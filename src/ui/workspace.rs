use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use gpui::{
    AnyElement, App, Bounds, Context, CursorStyle, DispatchPhase, Entity, EntityInputHandler,
    FocusHandle, Focusable, InputHandler, KeyDownEvent, Keystroke, MouseButton, MouseDownEvent,
    MouseMoveEvent, MouseUpEvent, Point, ScrollDelta, ScrollHandle, ScrollWheelEvent, ShapedLine,
    SharedString, StrikethroughStyle, TextAlign, TextInputConfiguration, TextRun, TouchPhase,
    UTF16Selection, UnderlineStyle, Window, WindowControlArea, anchored, canvas, deferred, div,
    fill, font, outline, point, prelude::*, px, relative, rgb, rgba, size,
};

use crate::agent::AgentKind;
use crate::app::model::{AgentDump, PaneTreeDump, TabDump, WorkspaceDump};
use crate::app::{CommandTransport, ModelSnapshot};
use crate::command::{
    AppCommand, FocusDirection, OperationResult, PaneCommand, SplitDirection, TabCommand,
    TerminalCommand, WorkspaceCommand,
};
use crate::config::{AppConfig, DEFAULT_TERMINAL_LINE_HEIGHT, ThemeColors};
use crate::ids::{ConnectionId, PaneId, TabId, TerminalId, WorkspaceId};
use crate::pane::SplitAxis;
use crate::surface::SurfaceState;
use crate::terminal::{
    TerminalCell, TerminalColor, TerminalModes, TerminalRowSnapshot, TerminalSize, TerminalSnapshot,
};

use super::application::{
    ActivateTab1, ActivateTab2, ActivateTab3, ActivateTab4, ActivateTab5, ActivateTab6,
    ActivateTab7, ActivateTab8, ActivateTab9, ActivateTab10, HideWindow, IgnoreQuit,
    MinimizeWindow, NewTerminalTab, NewWorkspace, NextTab, NextWorkspace, PreviousTab,
    PreviousWorkspace, RenameTab, RenameWorkspace, SplitDown, SplitRight, ToggleSidebar,
    WaterApplication, shortcut_matches_or_default,
};

const DEFAULT_TERMINAL_CELL_WIDTH: f32 = 8.4;
/// Pixel step for a single click on the tab-strip overflow indicators.
const TAB_SCROLL_NUDGE_PX: f32 = 160.0;
/// Width of the sidebar disclosure column. Agent rows indent by the same
/// amount so nested rows line up with their workspace title.
const SIDEBAR_DISCLOSURE_WIDTH: f32 = 18.0;
/// Scrim painted behind in-window dialogs; darker than the old 60%
/// overlay so the background reads as inactive while a dialog is open.
const DIALOG_SCRIM: u32 = 0x000000e6;
/// Movement needed before a row click becomes a drag gesture.
const SIDEBAR_DRAG_THRESHOLD_PX: f32 = 4.0;
/// A drop boundary is active only near a group edge, not throughout a row.
const SIDEBAR_DROP_TOLERANCE_PX: f32 = 14.0;
/// Distance from the sidebar viewport edge that starts event-driven scrolling.
const SIDEBAR_AUTOSCROLL_EDGE_PX: f32 = 24.0;
/// Pixels to move the sidebar per captured pointer move near an edge.
const SIDEBAR_AUTOSCROLL_STEP_PX: f32 = 24.0;
const SPLIT_DIVIDER_WIDTH_PX: f32 = 6.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum WorkspaceConnectionKind {
    Local,
    Remote,
}

#[derive(Clone)]
pub(crate) struct WorkspaceConnection {
    pub(crate) id: ConnectionId,
    pub(crate) title: String,
    pub(crate) kind: WorkspaceConnectionKind,
    pub(crate) client: Arc<dyn CommandTransport>,
    pub(crate) snapshot: ModelSnapshot,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct TerminalMetrics {
    cell_width: f32,
    line_height: f32,
    scale_factor: f32,
}

impl Default for TerminalMetrics {
    fn default() -> Self {
        Self {
            cell_width: DEFAULT_TERMINAL_CELL_WIDTH,
            line_height: DEFAULT_TERMINAL_LINE_HEIGHT,
            scale_factor: 1.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct TerminalCellPosition {
    /// Viewport-relative row. Signed so a selected cell can remain tracked
    /// while scrolling moves it outside the visible area.
    row: i32,
    column: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TerminalSelectionSide {
    Left,
    Right,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RenameTarget {
    Tab(TabId),
    Agent {
        connection_id: ConnectionId,
        pane_id: PaneId,
    },
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct ContextMenuState {
    target: ContextMenuTarget,
    position: Point<gpui::Pixels>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ContextMenuTarget {
    Connection(ConnectionId),
    Workspace {
        connection_id: ConnectionId,
        workspace_id: WorkspaceId,
    },
    Tab(TabId),
    Agent {
        connection_id: ConnectionId,
        pane_id: PaneId,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SidebarDragSource {
    Workspace(WorkspaceId),
    Agent {
        pane_id: PaneId,
        workspace_id: WorkspaceId,
    },
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct SidebarDrag {
    source: SidebarDragSource,
    start: Point<gpui::Pixels>,
    active: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum SidebarDropPreview {
    Workspace {
        y: f32,
        index: usize,
    },
    Agent {
        y: f32,
        target_workspace_id: WorkspaceId,
    },
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct SplitRect {
    origin: f32,
    extent: f32,
}

#[derive(Debug, Clone, PartialEq)]
struct SplitDrag {
    tab_id: TabId,
    path: Vec<bool>,
    axis: SplitAxis,
    rect: SplitRect,
    start: f32,
    start_ratio: f32,
    preview_ratio: Option<f32>,
}

type SplitBounds = Arc<Mutex<BTreeMap<(TabId, Vec<bool>), SplitRect>>>;

/// Wakes a polled future once `deadline` passes. GPUI's pinned scheduler has
/// no sleep primitive, so the timer runs on a short-lived thread that hands
/// the wakeup through an mpsc channel; the channel is created lazily on the
/// first poll, so a future that is ready (or dropped) never spawns a thread.
struct CaretSleep {
    deadline: std::time::Instant,
    notified: Option<std::sync::mpsc::Receiver<()>>,
}

impl CaretSleep {
    fn after(duration: std::time::Duration) -> Self {
        Self {
            deadline: std::time::Instant::now() + duration,
            notified: None,
        }
    }
}

impl std::future::Future for CaretSleep {
    type Output = ();

    fn poll(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<()> {
        if std::time::Instant::now() >= self.deadline {
            return std::task::Poll::Ready(());
        }
        let deadline = self.deadline;
        let receiver = self.notified.get_or_insert_with(|| {
            let (sender, receiver) = std::sync::mpsc::channel();
            let _ = std::thread::Builder::new()
                .name("water-dialog-caret".to_owned())
                .spawn(move || {
                    let delay = deadline.saturating_duration_since(std::time::Instant::now());
                    std::thread::sleep(delay);
                    let _ = sender.send(());
                });
            receiver
        });
        match receiver.try_recv() {
            Ok(()) => {}
            Err(std::sync::mpsc::TryRecvError::Empty) => {}
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                return std::task::Poll::Ready(());
            }
        }
        cx.waker().wake_by_ref();
        std::task::Poll::Pending
    }
}

/// In-window dialogs are kept as view state so their presentation and
/// dismissal share one path without introducing model-owned UI state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DialogState {
    ConfirmCloseWorkspace {
        workspace_id: WorkspaceId,
    },
    ConnectRemote,
    RenameWorkspace {
        connection_id: ConnectionId,
        workspace_id: WorkspaceId,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TerminalSelectionEndpoint {
    position: TerminalCellPosition,
    side: TerminalSelectionSide,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TerminalSelection {
    terminal_id: TerminalId,
    anchor: TerminalSelectionEndpoint,
    head: TerminalSelectionEndpoint,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TerminalMouseReportKind {
    Press,
    Release,
    Motion,
}

#[derive(Debug, Clone, Copy)]
struct TerminalMouseContext {
    modes: TerminalModes,
    bounds: Option<Bounds<gpui::Pixels>>,
    metrics: TerminalMetrics,
}

#[derive(Debug, Clone, Copy)]
struct TerminalMousePosition {
    column: usize,
    row: usize,
    side: TerminalSelectionSide,
}

#[derive(Debug, Clone, Copy)]
struct TerminalRenderOptions {
    metrics: TerminalMetrics,
    theme: ThemeColors,
    cursor_focused: bool,
    /// UI-local normal-screen viewport movement relative to the latest
    /// snapshot. Positive values reveal `rows_before`; negative values reveal
    /// `rows_after`.
    scroll_offset_rows: f32,
}

/// Keeps wheel movement visually anchored while viewport requests cross the
/// asynchronous model/PTY boundary.
#[derive(Debug, Clone)]
struct TerminalScrollState {
    /// Worker viewport coordinate represented by the latest snapshot.
    observed_viewport_position: i64,
    /// Continuous UI-local movement not yet incorporated into the snapshot.
    visual_unacked_rows: f32,
    /// Latest absolute worker viewport target requested by the UI.
    requested_viewport_position: i64,
    /// Start of the latest absolute request, used only by opt-in scroll stats.
    request_started_at: Option<Instant>,
}

const MOUSE_SCROLL_ANIMATION_DURATION: Duration = Duration::from_millis(72);
const MOUSE_SCROLL_IMMEDIATE_FRACTION: f32 = 0.2;
const PREPARED_ROW_LOOKAHEAD: i32 = 8;
const TERMINAL_SELECTION_AUTOSCROLL_MARGIN_PX: f32 = 24.0;
const TERMINAL_SELECTION_AUTOSCROLL_STEP_ROWS: i64 = 3;

#[derive(Debug, Clone)]
struct TerminalMouseScrollAnimation {
    pane_id: PaneId,
    start_position: f32,
    target_position: f32,
    started_at: Instant,
}

#[derive(Debug, Clone, Copy)]
struct TerminalSelectionAutoscroll {
    terminal_id: TerminalId,
    position: Point<gpui::Pixels>,
}

impl TerminalMouseScrollAnimation {
    fn position_at(&self, now: Instant) -> (f32, bool) {
        let elapsed = now.saturating_duration_since(self.started_at);
        let progress =
            (elapsed.as_secs_f32() / MOUSE_SCROLL_ANIMATION_DURATION.as_secs_f32()).clamp(0.0, 1.0);
        // A time-based ease-out reaches the same point after a dropped frame;
        // it never counts frames or assumes a fixed 16 ms refresh interval.
        let eased = 1.0 - (1.0 - progress).powi(3);
        (
            self.start_position + (self.target_position - self.start_position) * eased,
            progress >= 1.0,
        )
    }
}

impl TerminalScrollState {
    fn new(viewport_position: i64) -> Self {
        Self {
            observed_viewport_position: viewport_position,
            visual_unacked_rows: 0.0,
            requested_viewport_position: viewport_position,
            request_started_at: None,
        }
    }

    #[cfg(test)]
    fn accumulate(&mut self, delta_rows: f32) -> (Option<i64>, bool) {
        self.accumulate_with_boundaries(delta_rows, false, false)
    }

    fn accumulate_with_boundaries(
        &mut self,
        delta_rows: f32,
        at_history_start: bool,
        at_live_bottom: bool,
    ) -> (Option<i64>, bool) {
        if !delta_rows.is_finite() || delta_rows == 0.0 {
            return (None, false);
        }

        let had_boundary_debt = (at_history_start && self.visual_unacked_rows > 0.0)
            || (at_live_bottom && self.visual_unacked_rows < 0.0);
        if had_boundary_debt {
            self.visual_unacked_rows = 0.0;
            self.requested_viewport_position = self.observed_viewport_position;
            self.request_started_at = None;
        }

        let delta_rows = delta_rows.clamp(-100.0, 100.0);
        let previous_visual = self.visual_unacked_rows;
        self.visual_unacked_rows = (previous_visual + delta_rows).clamp(-101.0, 101.0);
        // A known terminal boundary is a hard physical limit, not merely a
        // paint clamp. Discard movement beyond it immediately so reversing
        // direction never has to repay invisible accumulated wheel deltas.
        if at_history_start {
            self.visual_unacked_rows = self.visual_unacked_rows.min(0.0);
        }
        if at_live_bottom {
            self.visual_unacked_rows = self.visual_unacked_rows.max(0.0);
        }
        let whole = self.visual_unacked_rows.trunc() as i64;
        let target = self.observed_viewport_position.saturating_add(whole);
        let request = (target != self.requested_viewport_position).then_some(target);
        self.requested_viewport_position = target;
        if request.is_some() {
            self.request_started_at = Some(Instant::now());
            scroll_stat_inc(&SCROLL_VIEWPORT_REQUESTS);
        }
        scroll_stat_max_unacked(self.visual_unacked_rows);
        (request, self.visual_unacked_rows != previous_visual)
    }
}

fn reconcile_visual_scroll(state: &mut TerminalScrollState, snapshot: &TerminalSnapshot) {
    let applied = snapshot
        .viewport_position
        .saturating_sub(state.observed_viewport_position);
    state.observed_viewport_position = snapshot.viewport_position;
    state.visual_unacked_rows -= applied as f32;
    if applied != 0 {
        scroll_stat_inc(&SCROLL_VIEWPORT_ACKS);
        if scroll_stats_enabled()
            && let Some(started) = state.request_started_at.take()
        {
            SCROLL_ACK_LATENCY_MICROS
                .fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
        }
    }

    // Clamp only at a known history boundary. An acknowledgement always
    // rebases first so replacing the snapshot cannot move the visual viewport.
    if (state.visual_unacked_rows > 0.0 && snapshot.rows_before.is_empty())
        || (state.visual_unacked_rows < 0.0 && snapshot.rows_after.is_empty())
    {
        state.visual_unacked_rows = 0.0;
        state.requested_viewport_position = snapshot.viewport_position;
    }
}

fn record_latest_viewport_request(
    pending: &mut BTreeMap<TerminalId, (PaneId, i64)>,
    terminal_id: TerminalId,
    pane_id: PaneId,
    target: i64,
) {
    pending.insert(terminal_id, (pane_id, target));
}

static SCROLL_STATS_ENABLED: OnceLock<bool> = OnceLock::new();
static SCROLL_STATS_REPORTED: AtomicBool = AtomicBool::new(false);
static SCROLL_WHEEL_EVENTS: AtomicU64 = AtomicU64::new(0);
static SCROLL_TRACKPAD_EVENTS: AtomicU64 = AtomicU64::new(0);
static SCROLL_MOUSE_EVENTS: AtomicU64 = AtomicU64::new(0);
static SCROLL_VIEWPORT_REQUESTS: AtomicU64 = AtomicU64::new(0);
static SCROLL_VIEWPORT_ACKS: AtomicU64 = AtomicU64::new(0);
static SCROLL_ACK_LATENCY_MICROS: AtomicU64 = AtomicU64::new(0);
static SCROLL_MAX_UNACKED_MILLIROWS: AtomicU64 = AtomicU64::new(0);
static SCROLL_PREPAINT_MICROS: AtomicU64 = AtomicU64::new(0);
static SCROLL_SHAPE_LINE_COUNT: AtomicU64 = AtomicU64::new(0);
static SCROLL_FRAMES: AtomicU64 = AtomicU64::new(0);

fn scroll_stats_enabled() -> bool {
    *SCROLL_STATS_ENABLED.get_or_init(|| std::env::var_os("WATER_SCROLL_STATS").is_some())
}

fn scroll_stat_inc(counter: &AtomicU64) {
    if scroll_stats_enabled() {
        counter.fetch_add(1, Ordering::Relaxed);
    }
}

fn scroll_stat_max_unacked(rows: f32) {
    if scroll_stats_enabled() {
        SCROLL_MAX_UNACKED_MILLIROWS
            .fetch_max((rows.abs() * 1_000.0).round() as u64, Ordering::Relaxed);
    }
}

fn report_scroll_stats_if_enabled() {
    if !scroll_stats_enabled() || SCROLL_STATS_REPORTED.swap(true, Ordering::Relaxed) {
        return;
    }
    let acks = SCROLL_VIEWPORT_ACKS.load(Ordering::Relaxed);
    let ack_micros = SCROLL_ACK_LATENCY_MICROS.load(Ordering::Relaxed);
    tracing::info!(
        target: "water::scroll",
        wheel_events = SCROLL_WHEEL_EVENTS.load(Ordering::Relaxed),
        trackpad_events = SCROLL_TRACKPAD_EVENTS.load(Ordering::Relaxed),
        mouse_events = SCROLL_MOUSE_EVENTS.load(Ordering::Relaxed),
        viewport_requests = SCROLL_VIEWPORT_REQUESTS.load(Ordering::Relaxed),
        viewport_acks = acks,
        max_unacked_rows = SCROLL_MAX_UNACKED_MILLIROWS.load(Ordering::Relaxed) as f64 / 1_000.0,
        viewport_ack_latency_us = ack_micros.checked_div(acks).unwrap_or(0),
        terminal_prepaint_us = SCROLL_PREPAINT_MICROS.load(Ordering::Relaxed),
        shape_line_count = SCROLL_SHAPE_LINE_COUNT.load(Ordering::Relaxed),
        scroll_frames = SCROLL_FRAMES.load(Ordering::Relaxed),
        "terminal scroll stats"
    );
}

#[derive(Debug, Clone, Copy)]
struct TerminalBackgroundSpan {
    start_column: usize,
    width_columns: usize,
    color: u32,
}

struct TerminalTextCell {
    text: String,
    run: TextRun,
    width_columns: usize,
}

struct TerminalTextChunk {
    start_column: usize,
    width_columns: usize,
    span_columns: usize,
    requires_cell_scaling: bool,
    text: String,
    runs: Vec<TextRun>,
    cells: Vec<TerminalTextCell>,
}

#[derive(Clone)]
struct TerminalTextPaint {
    start_column: usize,
    line: ShapedLine,
}

#[derive(Clone)]
struct TerminalRowPaint {
    /// Signed viewport-relative row. `-1` and `size.lines` are the optional
    /// overscan rows surrounding the visible grid.
    row: i32,
    text: Vec<TerminalTextPaint>,
    backgrounds: Vec<TerminalBackgroundSpan>,
}

struct TerminalPrepaintState {
    rows: Vec<TerminalRowPaint>,
    ime_line: Option<(ShapedLine, usize, usize)>,
}

#[derive(Clone, PartialEq)]
struct TerminalRenderCacheKey {
    terminal_id: TerminalId,
    snapshot_revision: u64,
    viewport_position: i64,
    /// The focused cursor is baked into its row's colors. Keep its position
    /// out of whole-cache compatibility, then invalidate only the old/new
    /// cursor rows when it moves.
    focused_cursor: Option<(usize, usize)>,
    font_family: String,
    font_size_bits: u32,
    metrics: TerminalMetrics,
    theme: ThemeColors,
    cursor_focused: bool,
    selection: Option<TerminalSelection>,
    bounds_origin_x_bits: u32,
    bounds_width_bits: u32,
}

impl TerminalRenderCacheKey {
    fn rows_compatible_with(&self, other: &Self) -> bool {
        self.terminal_id == other.terminal_id
            && self.font_family == other.font_family
            && self.font_size_bits == other.font_size_bits
            && self.metrics == other.metrics
            && self.theme == other.theme
            && self.cursor_focused == other.cursor_focused
            && self.selection == other.selection
            && self.bounds_origin_x_bits == other.bounds_origin_x_bits
            && self.bounds_width_bits == other.bounds_width_bits
    }
}

struct TerminalCachedRowPaint {
    cells: TerminalRowSnapshot,
    paint: TerminalRowPaint,
}

#[derive(Default)]
struct TerminalRenderCache {
    key: Option<TerminalRenderCacheKey>,
    rows: BTreeMap<i32, TerminalCachedRowPaint>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TerminalResizeRequest {
    terminal_id: TerminalId,
    target: TerminalSize,
    /// Size projected by the snapshot when this request was sent. If this
    /// changes while the target does not, another window has superseded us
    /// and the active window must be allowed to request its target again.
    observed_size: TerminalSize,
}

type TerminalRenderCaches = Arc<Mutex<BTreeMap<TerminalId, TerminalRenderCache>>>;

struct TerminalRenderElement {
    /// Shared with the terminal registry; painting never copies the grid.
    snapshot: Arc<TerminalSnapshot>,
    selection: Option<TerminalSelection>,
    options: TerminalRenderOptions,
    font_family: String,
    font_size: f32,
    ime_text: Option<String>,
    terminal_bounds: Arc<Mutex<BTreeMap<TerminalId, Bounds<gpui::Pixels>>>>,
    render_caches: TerminalRenderCaches,
    input_handler: Option<(Entity<WorkspaceView>, FocusHandle)>,
}

struct TerminalInputHandler {
    view: Entity<WorkspaceView>,
    terminal_id: TerminalId,
    element_bounds: Bounds<gpui::Pixels>,
    cursor: (usize, usize),
}

impl TerminalInputHandler {
    fn update_view<R>(
        &self,
        cx: &mut App,
        update: impl FnOnce(&mut WorkspaceView, &mut Context<WorkspaceView>) -> R,
    ) -> R {
        let fallback_terminal_id = self.terminal_id;
        self.view.update(cx, |view, cx| {
            let terminal_id = view.active_terminal_id().unwrap_or(fallback_terminal_id);
            view.ime_terminal = Some(terminal_id);
            update(view, cx)
        })
    }
}

impl InputHandler for TerminalInputHandler {
    fn selected_text_range(
        &mut self,
        ignore_disabled_input: bool,
        window: &mut Window,
        cx: &mut App,
    ) -> Option<UTF16Selection> {
        self.update_view(cx, |view, cx| {
            <WorkspaceView as EntityInputHandler>::selected_text_range(
                view,
                ignore_disabled_input,
                window,
                cx,
            )
        })
    }

    fn marked_text_range(&mut self, window: &mut Window, cx: &mut App) -> Option<Range<usize>> {
        self.update_view(cx, |view, cx| {
            <WorkspaceView as EntityInputHandler>::marked_text_range(view, window, cx)
        })
    }

    fn text_for_range(
        &mut self,
        range_utf16: Range<usize>,
        adjusted_range: &mut Option<Range<usize>>,
        window: &mut Window,
        cx: &mut App,
    ) -> Option<String> {
        self.update_view(cx, |view, cx| {
            <WorkspaceView as EntityInputHandler>::text_for_range(
                view,
                range_utf16,
                adjusted_range,
                window,
                cx,
            )
        })
    }

    fn replace_text_in_range(
        &mut self,
        replacement_range: Option<Range<usize>>,
        text: &str,
        window: &mut Window,
        cx: &mut App,
    ) {
        self.update_view(cx, |view, cx| {
            <WorkspaceView as EntityInputHandler>::replace_text_in_range(
                view,
                replacement_range,
                text,
                window,
                cx,
            )
        });
    }

    fn replace_and_mark_text_in_range(
        &mut self,
        range_utf16: Option<Range<usize>>,
        new_text: &str,
        new_selected_range: Option<Range<usize>>,
        window: &mut Window,
        cx: &mut App,
    ) {
        self.update_view(cx, |view, cx| {
            <WorkspaceView as EntityInputHandler>::replace_and_mark_text_in_range(
                view,
                range_utf16,
                new_text,
                new_selected_range,
                window,
                cx,
            )
        });
    }

    fn unmark_text(&mut self, window: &mut Window, cx: &mut App) {
        self.update_view(cx, |view, cx| {
            <WorkspaceView as EntityInputHandler>::unmark_text(view, window, cx)
        });
    }

    fn bounds_for_range(
        &mut self,
        _range_utf16: Range<usize>,
        _window: &mut Window,
        cx: &mut App,
    ) -> Option<Bounds<gpui::Pixels>> {
        let element_bounds = self.element_bounds;
        let fallback_terminal_id = self.terminal_id;
        let fallback_cursor = self.cursor;
        self.update_view(cx, |view, _cx| {
            let terminal_id = view.active_terminal_id().unwrap_or(fallback_terminal_id);
            let bounds = view
                .terminal_bounds_for(terminal_id)
                .unwrap_or(element_bounds);
            let (cursor, width_columns) = view
                .terminal_snapshot_for(terminal_id)
                .map(|snapshot| {
                    let cursor = terminal_cursor_position(snapshot);
                    let width_columns = snapshot
                        .cell(cursor.0, cursor.1)
                        .map(|cell| if cell.flags.wide() { 2 } else { 1 })
                        .unwrap_or(1);
                    (cursor, width_columns)
                })
                .unwrap_or((fallback_cursor, 1));
            Some(terminal_cell_bounds(
                bounds,
                view.terminal_metrics,
                cursor.0,
                cursor.1,
                width_columns,
            ))
        })
    }

    fn character_index_for_point(
        &mut self,
        point: Point<gpui::Pixels>,
        window: &mut Window,
        cx: &mut App,
    ) -> Option<usize> {
        self.update_view(cx, |view, cx| {
            <WorkspaceView as EntityInputHandler>::character_index_for_point(
                view, point, window, cx,
            )
        })
    }

    fn set_selected_text_range(
        &mut self,
        range_utf16: Range<usize>,
        window: &mut Window,
        cx: &mut App,
    ) {
        self.update_view(cx, |view, cx| {
            <WorkspaceView as EntityInputHandler>::set_selected_text_range(
                view,
                range_utf16,
                window,
                cx,
            )
        });
    }

    fn element_bounds(
        &mut self,
        _window: &mut Window,
        _cx: &mut App,
    ) -> Option<Bounds<gpui::Pixels>> {
        Some(self.element_bounds)
    }

    fn text_length_utf16(&mut self, window: &mut Window, cx: &mut App) -> Option<usize> {
        self.update_view(cx, |view, cx| {
            <WorkspaceView as EntityInputHandler>::text_length_utf16(view, window, cx)
        })
    }

    fn apple_press_and_hold_enabled(&mut self) -> bool {
        false
    }

    fn accepts_text_input(&mut self, _window: &mut Window, cx: &mut App) -> bool {
        let view = self.view.read(cx);
        view.active_terminal_id().is_some() || view.ime_terminal == Some(self.terminal_id)
    }

    fn prefers_ime_for_printable_keys(&mut self, _window: &mut Window, cx: &mut App) -> bool {
        let view = self.view.read(cx);
        view.active_terminal_id().is_some() || view.ime_terminal == Some(self.terminal_id)
    }

    fn text_input_configuration(
        &mut self,
        _window: &mut Window,
        _cx: &mut App,
    ) -> TextInputConfiguration {
        TextInputConfiguration::default()
    }

    fn text_input_editable_range(
        &mut self,
        window: &mut Window,
        cx: &mut App,
    ) -> Option<Range<usize>> {
        self.update_view(cx, |view, cx| {
            <WorkspaceView as EntityInputHandler>::text_input_editable_range(view, window, cx)
        })
    }
}

pub struct WorkspaceView {
    application: Option<WaterApplication>,
    connections: Vec<WorkspaceConnection>,
    active_connection: ConnectionId,
    client: std::sync::Arc<dyn CommandTransport>,
    snapshot: ModelSnapshot,
    config: AppConfig,
    terminal_metrics: TerminalMetrics,
    focus_handle: FocusHandle,
    resize_requests: Arc<Mutex<BTreeMap<(ConnectionId, PaneId), TerminalResizeRequest>>>,
    terminal_bounds: Arc<Mutex<BTreeMap<TerminalId, Bounds<gpui::Pixels>>>>,
    render_caches: TerminalRenderCaches,
    input_handler_terminal: Option<TerminalId>,
    /// Workspace selection and pane focus belong to this projection. The
    /// model's active aliases are compatibility state shared by all windows.
    selected_workspace: Option<WorkspaceId>,
    focused_pane: Option<PaneId>,
    scroll_accumulators: BTreeMap<TerminalId, TerminalScrollState>,
    /// Locally rendered terminal snapshots, built from this connection's
    /// local emulators (the server projects control-plane metadata only).
    terminal_snapshots: BTreeMap<TerminalId, std::sync::Arc<TerminalSnapshot>>,
    active_trackpad_scrolls: BTreeSet<TerminalId>,
    pending_viewport_requests: BTreeMap<TerminalId, (PaneId, i64)>,
    viewport_request_frame_pending: bool,
    mouse_scroll_animations: BTreeMap<TerminalId, TerminalMouseScrollAnimation>,
    mouse_scroll_frame_pending: bool,
    selection_autoscroll: Option<TerminalSelectionAutoscroll>,
    selection_autoscroll_frame_pending: bool,
    selection: Option<TerminalSelection>,
    ime_terminal: Option<TerminalId>,
    ime_marked_text: String,
    ime_selected_range: Range<usize>,
    dragging_terminal: Option<TerminalId>,
    reported_mouse: Option<(TerminalId, MouseButton)>,
    last_reported_mouse_cell: Option<(TerminalId, TerminalCellPosition)>,
    sidebar_collapsed: bool,
    collapsed_connections: BTreeSet<ConnectionId>,
    collapsed_workspaces: BTreeSet<(ConnectionId, WorkspaceId)>,
    sidebar_scroll: ScrollHandle,
    tab_scroll: ScrollHandle,
    /// Tab-strip shape seen at the last snapshot install; changes trigger a
    /// follow-up render so the overflow indicators pick up the fresh layout.
    last_tab_strip_signature: (Option<WorkspaceId>, usize),
    /// Tab whose selection the strip last revealed (auto-scroll on switch).
    last_revealed_tab: Option<TabId>,
    /// Optimistic tab target for relative tab navigation (Cmd-[ / Cmd-])
    /// while an activation command is still in flight.
    pending_tab: Option<TabId>,
    sidebar_width: f32,
    dragging_sidebar: bool,
    window_drag_start: Option<Point<gpui::Pixels>>,
    sidebar_drag: Option<SidebarDrag>,
    sidebar_drop_preview: Option<SidebarDropPreview>,
    split_bounds: SplitBounds,
    split_drag: Option<SplitDrag>,
    titlebar_dragging: bool,
    rename_target: Option<RenameTarget>,
    rename_value: String,
    /// Shared editable value for the text-input dialogs (connect remote,
    /// rename workspace). `dialog_caret` is a byte offset into `dialog_input`.
    dialog_input: String,
    dialog_caret: usize,
    dialog_caret_visible: bool,
    /// Byte offset in `dialog_input` where the current IME composition
    /// region starts; committed or cancelled through the EntityInputHandler
    /// overrides below while a text-input dialog is open.
    dialog_ime_base: usize,
    /// Bumped every time a text-input dialog opens; the caret blink task for
    /// a previous dialog notices the mismatch and stops.
    dialog_input_generation: u64,
    remote_connection_error: Option<String>,
    remote_connection_pending: bool,
    context_menu: Option<ContextMenuState>,
    dialog: Option<DialogState>,
}

impl Drop for WorkspaceView {
    fn drop(&mut self) {
        report_scroll_stats_if_enabled();
    }
}

impl WorkspaceView {
    pub fn new(
        client: std::sync::Arc<dyn CommandTransport>,
        snapshot: ModelSnapshot,
        focus_handle: FocusHandle,
    ) -> Self {
        Self::new_with_config(client, snapshot, focus_handle, AppConfig::default())
    }

    pub fn new_with_config(
        client: std::sync::Arc<dyn CommandTransport>,
        snapshot: ModelSnapshot,
        focus_handle: FocusHandle,
        config: AppConfig,
    ) -> Self {
        let connection_id = ConnectionId::new(1);
        Self::new_with_connections(
            None,
            vec![WorkspaceConnection {
                id: connection_id,
                title: "Local".to_owned(),
                kind: WorkspaceConnectionKind::Local,
                client,
                snapshot,
            }],
            connection_id,
            focus_handle,
            config,
        )
    }

    pub(crate) fn new_with_connections(
        application: Option<WaterApplication>,
        connections: Vec<WorkspaceConnection>,
        active_connection: ConnectionId,
        focus_handle: FocusHandle,
        config: AppConfig,
    ) -> Self {
        let config = config.normalized();
        let active = connections
            .iter()
            .find(|connection| connection.id == active_connection)
            .or_else(|| connections.first())
            .expect("workspace view requires at least one connection");
        let client = active.client.clone();
        let snapshot = active.snapshot.clone();
        let active_connection = active.id;
        let selected_workspace = workspace_selection_after_snapshot(None, &snapshot);
        let focused_pane =
            focused_pane_for_workspace(&snapshot, selected_workspace, snapshot.focused_pane);
        let sidebar_width = config.ui.sidebar_width;
        let sidebar_collapsed = !config.ui.sidebar_visible;
        Self {
            application,
            connections,
            active_connection,
            client,
            snapshot,
            config,
            terminal_metrics: TerminalMetrics::default(),
            focus_handle,
            resize_requests: Arc::new(Mutex::new(BTreeMap::new())),
            terminal_bounds: Arc::new(Mutex::new(BTreeMap::new())),
            render_caches: Arc::new(Mutex::new(BTreeMap::new())),
            input_handler_terminal: None,
            selected_workspace,
            focused_pane,
            scroll_accumulators: BTreeMap::new(),
            active_trackpad_scrolls: BTreeSet::new(),
            pending_viewport_requests: BTreeMap::new(),
            viewport_request_frame_pending: false,
            mouse_scroll_animations: BTreeMap::new(),
            mouse_scroll_frame_pending: false,
            selection_autoscroll: None,
            selection_autoscroll_frame_pending: false,
            selection: None,
            ime_terminal: None,
            ime_marked_text: String::new(),
            ime_selected_range: 0..0,
            dragging_terminal: None,
            reported_mouse: None,
            last_reported_mouse_cell: None,
            sidebar_collapsed,
            terminal_snapshots: BTreeMap::new(),
            collapsed_connections: BTreeSet::new(),
            collapsed_workspaces: BTreeSet::new(),
            sidebar_scroll: ScrollHandle::new(),
            tab_scroll: ScrollHandle::new(),
            last_tab_strip_signature: (None, 0),
            last_revealed_tab: None,
            pending_tab: None,
            sidebar_width,
            dragging_sidebar: false,
            window_drag_start: None,
            sidebar_drag: None,
            sidebar_drop_preview: None,
            split_bounds: Arc::new(Mutex::new(BTreeMap::new())),
            split_drag: None,
            titlebar_dragging: false,
            rename_target: None,
            rename_value: String::new(),
            dialog_input: String::new(),
            dialog_caret: 0,
            dialog_caret_visible: true,
            dialog_ime_base: 0,
            dialog_input_generation: 0,
            remote_connection_error: None,
            remote_connection_pending: false,
            context_menu: None,
            dialog: None,
        }
    }

    /// Returns this window's workspace projection, never the process-global
    /// compatibility alias from the model snapshot.
    fn selected_workspace_dump(&self) -> Option<&WorkspaceDump> {
        self.selected_workspace
            .and_then(|workspace_id| self.workspace_by_id(workspace_id))
    }

    fn active_workspace_id(&self) -> Option<WorkspaceId> {
        self.selected_workspace
    }

    fn connection_by_id(&self, connection_id: ConnectionId) -> Option<&WorkspaceConnection> {
        self.connections
            .iter()
            .find(|connection| connection.id == connection_id)
    }

    fn reset_active_connection_projection(&mut self, connection: WorkspaceConnection) {
        self.active_connection = connection.id;
        self.client = connection.client;
        self.snapshot = connection.snapshot;
        self.selected_workspace = workspace_selection_after_snapshot(None, &self.snapshot);
        self.focused_pane = focused_pane_for_workspace(
            &self.snapshot,
            self.selected_workspace,
            self.snapshot.focused_pane,
        );
        self.scroll_accumulators.clear();
        self.active_trackpad_scrolls.clear();
        self.pending_viewport_requests.clear();
        self.mouse_scroll_animations.clear();
        self.render_caches
            .lock()
            .expect("terminal render caches poisoned")
            .clear();
        self.pending_tab = None;
        self.last_revealed_tab = None;
        self.split_drag = None;
        self.selection = None;
        self.clear_ime();
    }

    fn select_connection_locally(
        &mut self,
        connection_id: ConnectionId,
        cx: &mut Context<Self>,
    ) -> bool {
        if self.active_connection == connection_id {
            return false;
        }
        let Some(connection) = self.connection_by_id(connection_id).cloned() else {
            return false;
        };
        self.reset_active_connection_projection(connection);
        cx.notify();
        true
    }

    pub(crate) fn install_connection(
        &mut self,
        connection: WorkspaceConnection,
        cx: &mut Context<Self>,
    ) {
        if let Some(existing) = self
            .connections
            .iter_mut()
            .find(|existing| existing.id == connection.id)
        {
            *existing = connection;
        } else {
            self.connections.push(connection);
        }
        cx.notify();
    }

    pub(crate) fn install_connection_snapshot(
        &mut self,
        connection_id: ConnectionId,
        snapshot: ModelSnapshot,
        cx: &mut Context<Self>,
    ) {
        if connection_id == self.active_connection {
            self.install_snapshot(snapshot, cx);
            return;
        }
        let Some(connection) = self
            .connections
            .iter_mut()
            .find(|connection| connection.id == connection_id)
        else {
            return;
        };
        if snapshot.state_revision > connection.snapshot.state_revision {
            connection.snapshot = snapshot;
            cx.notify();
        }
    }

    pub(crate) fn remove_connection(
        &mut self,
        connection_id: ConnectionId,
        cx: &mut Context<Self>,
    ) {
        self.connections
            .retain(|connection| connection.id != connection_id);
        self.collapsed_connections.remove(&connection_id);
        self.collapsed_workspaces
            .retain(|(id, _)| *id != connection_id);
        if self.active_connection == connection_id
            && let Some(connection) = self.connections.first().cloned()
        {
            self.reset_active_connection_projection(connection);
        }
        self.context_menu = None;
        if self
            .dialog
            .is_some_and(|dialog| !self.dialog_target_exists(dialog))
        {
            self.dialog = None;
            self.clear_dialog_input();
        }
        cx.notify();
    }

    fn select_workspace_locally(
        &mut self,
        workspace_id: WorkspaceId,
        cx: &mut Context<Self>,
    ) -> bool {
        let Some(workspace) = self.workspace_by_id(workspace_id) else {
            return false;
        };
        let focused_pane = workspace_active_pane(workspace);
        let changed =
            self.selected_workspace != Some(workspace_id) || self.focused_pane != focused_pane;
        self.selected_workspace = Some(workspace_id);
        self.focused_pane = focused_pane;
        self.pending_tab = None;
        self.split_drag = None;
        self.selection = None;
        self.clear_ime();
        if changed {
            cx.notify();
        }
        changed
    }

    fn has_transient_ui(&self) -> bool {
        self.rename_target.is_some() || self.context_menu.is_some() || self.dialog.is_some()
    }

    fn toggle_sidebar(&mut self, cx: &mut Context<Self>) {
        self.sidebar_collapsed = !self.sidebar_collapsed;
        cx.notify();
    }

    fn toggle_connection_collapsed(&mut self, connection_id: ConnectionId, cx: &mut Context<Self>) {
        if self.connection_by_id(connection_id).is_none() {
            return;
        }
        if !self.collapsed_connections.remove(&connection_id) {
            self.collapsed_connections.insert(connection_id);
        }
        cx.notify();
    }

    fn toggle_workspace_collapsed(
        &mut self,
        connection_id: ConnectionId,
        workspace_id: WorkspaceId,
        cx: &mut Context<Self>,
    ) {
        let exists = self
            .connection_by_id(connection_id)
            .is_some_and(|connection| {
                workspace_exists_in_snapshot(&connection.snapshot, workspace_id)
            });
        if !exists {
            return;
        }
        let key = (connection_id, workspace_id);
        if !self.collapsed_workspaces.remove(&key) {
            self.collapsed_workspaces.insert(key);
        }
        cx.notify();
    }

    fn workspace_exists(&self, workspace_id: WorkspaceId) -> bool {
        workspace_exists_in_snapshot(&self.snapshot, workspace_id)
    }

    fn scroll_tab_bar(&mut self, event: &ScrollWheelEvent, cx: &mut Context<Self>) {
        // Horizontal wheel/trackpad deltas are scrolled by the container's
        // built-in handler (single application; see render_tab_bar). This
        // path only exists for the opt-in vertical gesture.
        if !self.config.ui.tab_bar_vertical_wheel_scroll {
            return;
        }
        let delta = event.delta.pixel_delta(px(24.));
        if f32::from(delta.x).abs() > f32::EPSILON {
            return;
        }
        let vertical = f32::from(delta.y);
        if vertical.abs() <= f32::EPSILON {
            return;
        }
        self.nudge_tab_bar_scroll(-vertical, cx);
    }

    /// Move the tab-strip scroll position by `delta_x` pixels, clamped to the
    /// overflow range that the last layout pass measured.
    fn nudge_tab_bar_scroll(&mut self, delta_x: f32, cx: &mut Context<Self>) {
        let offset = self.tab_scroll.offset();
        let max_offset = self.tab_scroll.max_offset();
        let next_x = (f32::from(offset.x) - delta_x).clamp(-f32::from(max_offset.x), 0.0);
        if (next_x - f32::from(offset.x)).abs() <= f32::EPSILON {
            return;
        }
        self.tab_scroll.set_offset(point(px(next_x), offset.y));
        cx.notify();
    }

    fn activate_tab_index(&mut self, index: usize, cx: &mut Context<Self>) {
        let Some(tab) = self
            .selected_workspace_dump()
            .and_then(|workspace| workspace.tabs.get(index))
        else {
            return;
        };
        let (tab_id, tab_active_pane) = (tab.id, tab.active_pane);
        self.pending_tab = Some(tab_id);
        self.split_drag = None;
        self.focused_pane = Some(tab_active_pane);
        self.selection = None;
        self.clear_ime();
        self.dispatch(
            AppCommand::Tab(TabCommand::Activate {
                tab_id: Some(tab_id),
                index: None,
            }),
            cx,
        );
    }

    fn activate_relative_tab(&mut self, direction: isize, cx: &mut Context<Self>) {
        let Some(workspace) = self.selected_workspace_dump() else {
            return;
        };
        if workspace.tabs.is_empty() {
            return;
        }
        let snapshot_index = workspace
            .active_tab
            .and_then(|tab_id| workspace.tabs.iter().position(|tab| tab.id == tab_id))
            .unwrap_or(0);
        // Keep advancing from the optimistic target while the previous
        // activation is still in flight, so rapid presses each move once.
        let active_index = self
            .pending_tab
            .and_then(|pending| workspace.tabs.iter().position(|tab| tab.id == pending))
            .unwrap_or(snapshot_index);
        let tab_count = workspace.tabs.len() as isize;
        let next_index = (active_index as isize + direction).rem_euclid(tab_count) as usize;
        self.activate_tab_index(next_index, cx);
    }

    fn activate_relative_workspace(&mut self, direction: isize, cx: &mut Context<Self>) {
        let workspaces = self.workspace_dumps();
        if workspaces.is_empty() {
            return;
        }
        let current_index = self
            .selected_workspace
            .and_then(|workspace_id| {
                workspaces
                    .iter()
                    .position(|workspace| workspace.id == workspace_id)
            })
            .unwrap_or(0);
        let workspace_count = workspaces.len() as isize;
        let next_index = (current_index as isize + direction).rem_euclid(workspace_count) as usize;
        let workspace_id = workspaces[next_index].id;
        // Apply the activation locally first: the command is idempotent and
        // its pushed snapshot confirms, but waiting for the round trip made
        // the switch feel dead on remote connections.
        if self.apply_workspace_activated_locally(workspace_id) {
            cx.notify();
        }
        self.dispatch(
            AppCommand::Workspace(WorkspaceCommand::Activate {
                workspace_id: Some(workspace_id),
            }),
            cx,
        );
    }

    pub(crate) fn apply_config(&mut self, config: AppConfig, cx: &mut Context<Self>) {
        let config = config.normalized();
        self.sidebar_width = config.ui.sidebar_width;
        self.sidebar_collapsed = !config.ui.sidebar_visible;
        self.config = config;
        cx.notify();
    }

    pub(crate) fn new_terminal_tab(&mut self, cx: &mut Context<Self>) {
        let Some(workspace_id) = self.selected_workspace else {
            return;
        };
        self.dispatch(
            AppCommand::Tab(TabCommand::NewInWorkspace {
                workspace_id,
                title: None,
            }),
            cx,
        );
    }

    pub(crate) fn split_active_pane(&mut self, direction: SplitDirection, cx: &mut Context<Self>) {
        if let Some(pane_id) = self.focused_pane
            && self
                .selected_workspace
                .is_some_and(|workspace_id| self.workspace_contains_pane(workspace_id, pane_id))
        {
            self.dispatch(
                AppCommand::Pane(PaneCommand::Split {
                    pane_id: Some(pane_id),
                    direction,
                }),
                cx,
            );
        }
    }

    fn update_sidebar_width(&mut self, x: gpui::Pixels, cx: &mut Context<Self>) {
        if !self.dragging_sidebar || self.sidebar_collapsed {
            return;
        }
        let width = f32::from(x).clamp(
            self.config.ui.sidebar_min_width,
            self.config.ui.sidebar_max_width,
        );
        if (self.sidebar_width - width).abs() > f32::EPSILON {
            self.sidebar_width = width;
            cx.notify();
        }
    }

    fn begin_window_drag(&mut self, position: Point<gpui::Pixels>) {
        if self.has_transient_ui() {
            return;
        }
        self.window_drag_start = Some(position);
        self.sidebar_drag = None;
        self.sidebar_drop_preview = None;
        self.split_drag = None;
    }

    fn update_window_drag(&mut self, position: Point<gpui::Pixels>) -> bool {
        let Some(start) = self.window_drag_start else {
            return false;
        };
        let dx = (f32::from(position.x) - f32::from(start.x)).abs();
        let dy = (f32::from(position.y) - f32::from(start.y)).abs();
        if dx.max(dy) < SIDEBAR_DRAG_THRESHOLD_PX {
            return false;
        }
        self.window_drag_start = None;
        true
    }

    fn begin_sidebar_drag(&mut self, source: SidebarDragSource, start: Point<gpui::Pixels>) {
        self.sidebar_drag = Some(SidebarDrag {
            source,
            start,
            active: false,
        });
        self.sidebar_drop_preview = None;
        self.window_drag_start = None;
        self.split_drag = None;
    }

    fn sidebar_group_bounds(&self) -> Vec<(WorkspaceId, SidebarGroupGeometry)> {
        let offset = self.sidebar_scroll.offset();
        self.workspace_dumps()
            .into_iter()
            .enumerate()
            .filter_map(|(index, workspace)| {
                let bounds = self.sidebar_scroll.bounds_for_item(index)?;
                Some((
                    workspace.id,
                    SidebarGroupGeometry {
                        top: f32::from(bounds.top()) + f32::from(offset.y),
                        bottom: f32::from(bounds.bottom()) + f32::from(offset.y),
                    },
                ))
            })
            .collect()
    }

    fn update_sidebar_autoscroll(&self, position: Point<gpui::Pixels>) {
        let viewport = self.sidebar_scroll.bounds();
        if viewport.size.height <= px(0.) || !viewport.contains(&position) {
            return;
        }
        let y = f32::from(position.y);
        let top = f32::from(viewport.top());
        let bottom = f32::from(viewport.bottom());
        let offset = self.sidebar_scroll.offset();
        let max_offset = f32::from(self.sidebar_scroll.max_offset().y).max(0.0);
        let next_y = if y <= top + SIDEBAR_AUTOSCROLL_EDGE_PX {
            (f32::from(offset.y) + SIDEBAR_AUTOSCROLL_STEP_PX).min(0.0)
        } else if y >= bottom - SIDEBAR_AUTOSCROLL_EDGE_PX {
            (f32::from(offset.y) - SIDEBAR_AUTOSCROLL_STEP_PX).max(-max_offset)
        } else {
            return;
        };
        if (next_y - f32::from(offset.y)).abs() > f32::EPSILON {
            self.sidebar_scroll.set_offset(point(offset.x, px(next_y)));
        }
    }

    fn update_sidebar_drag(
        &mut self,
        position: Point<gpui::Pixels>,
        cx: &mut Context<Self>,
    ) -> bool {
        let Some(mut drag) = self.sidebar_drag else {
            return false;
        };
        if !drag.active {
            let dx = (f32::from(position.x) - f32::from(drag.start.x)).abs();
            let dy = (f32::from(position.y) - f32::from(drag.start.y)).abs();
            if dx.max(dy) < SIDEBAR_DRAG_THRESHOLD_PX {
                return true;
            }
            drag.active = true;
            self.sidebar_drag = Some(drag);
        }

        self.update_sidebar_autoscroll(position);
        let viewport = self.sidebar_scroll.bounds();
        let groups = self.sidebar_group_bounds();
        let y = f32::from(position.y);
        self.sidebar_drop_preview = if viewport.contains(&position) {
            match drag.source {
                SidebarDragSource::Workspace(source_id) => {
                    let group_bounds = groups.iter().map(|(_, bounds)| *bounds).collect::<Vec<_>>();
                    sidebar_drop_boundary(
                        y,
                        &group_bounds,
                        f32::from(viewport.top()),
                        f32::from(viewport.bottom()),
                        SIDEBAR_DROP_TOLERANCE_PX,
                    )
                    .and_then(|(boundary, line_y)| {
                        let source_index = groups
                            .iter()
                            .position(|(workspace_id, _)| *workspace_id == source_id)?;
                        let index =
                            sidebar_reorder_final_index(source_index, boundary, groups.len());
                        // Requirement: positions that would not change the
                        // order (endpoints next to the dragged group, the
                        // sole workspace) show no line and move nothing.
                        (index != source_index)
                            .then_some(SidebarDropPreview::Workspace { y: line_y, index })
                    })
                }
                SidebarDragSource::Agent {
                    workspace_id: source_workspace_id,
                    ..
                } => groups
                    .iter()
                    .find(|(workspace_id, bounds)| {
                        *workspace_id != source_workspace_id && y >= bounds.top && y < bounds.bottom
                    })
                    .map(|(target_workspace_id, bounds)| SidebarDropPreview::Agent {
                        y: bounds.bottom,
                        target_workspace_id: *target_workspace_id,
                    }),
            }
        } else {
            None
        };
        cx.notify();
        true
    }

    fn finish_sidebar_drag(&mut self, cx: &mut Context<Self>) {
        let drag = self.sidebar_drag.take();
        let preview = self.sidebar_drop_preview.take();
        let Some(drag) = drag else {
            return;
        };
        if !drag.active {
            return;
        }
        match (drag.source, preview) {
            (
                SidebarDragSource::Workspace(workspace_id),
                Some(SidebarDropPreview::Workspace { index, .. }),
            ) if self.workspace_exists(workspace_id) => {
                self.dispatch(
                    AppCommand::Workspace(WorkspaceCommand::Reorder {
                        workspace_id: Some(workspace_id),
                        index,
                    }),
                    cx,
                );
            }
            (
                SidebarDragSource::Agent {
                    pane_id,
                    workspace_id: source_workspace_id,
                },
                Some(SidebarDropPreview::Agent {
                    target_workspace_id,
                    ..
                }),
            ) if source_workspace_id != target_workspace_id
                && self.workspace_exists(target_workspace_id)
                && self
                    .agent_by_pane_id(pane_id)
                    .is_some_and(|agent| agent.workspace_id == source_workspace_id) =>
            {
                self.dispatch(
                    AppCommand::Pane(PaneCommand::MoveToWorkspace {
                        pane_id: Some(pane_id),
                        workspace_id: target_workspace_id,
                    }),
                    cx,
                );
            }
            _ => {}
        }
        cx.notify();
    }

    fn split_rect_for(&self, tab_id: TabId, path: &[bool]) -> Option<SplitRect> {
        self.split_bounds
            .lock()
            .expect("split bounds poisoned")
            .get(&(tab_id, path.to_vec()))
            .copied()
    }

    fn begin_split_drag(
        &mut self,
        tab_id: TabId,
        path: Vec<bool>,
        axis: SplitAxis,
        position: Point<gpui::Pixels>,
        cx: &mut Context<Self>,
    ) {
        let Some(tab) = self.active_tab_by_id(tab_id) else {
            return;
        };
        let Some((current_axis, ratio, _, _)) = pane_split_at_path(&tab.tree, &path) else {
            return;
        };
        if current_axis != axis {
            return;
        }
        let Some(rect) = self.split_rect_for(tab_id, &path) else {
            return;
        };
        let start = split_pointer_coordinate(position, axis);
        self.split_drag = Some(SplitDrag {
            tab_id,
            path,
            axis,
            rect,
            start,
            start_ratio: ratio,
            preview_ratio: None,
        });
        self.sidebar_drag = None;
        self.sidebar_drop_preview = None;
        self.window_drag_start = None;
        cx.notify();
    }

    fn update_split_drag(&mut self, position: Point<gpui::Pixels>, cx: &mut Context<Self>) -> bool {
        let Some(mut drag) = self.split_drag.take() else {
            return false;
        };
        let valid = self
            .active_tab_by_id(drag.tab_id)
            .and_then(|tab| pane_split_at_path(&tab.tree, &drag.path))
            .is_some_and(|(axis, _, _, _)| axis == drag.axis);
        if valid {
            drag.preview_ratio = Some(split_ratio_for_pointer(
                split_pointer_coordinate(position, drag.axis),
                drag.rect.origin,
                drag.rect.extent,
                SPLIT_DIVIDER_WIDTH_PX,
            ));
        }
        self.split_drag = Some(drag);
        cx.notify();
        true
    }

    fn finish_split_drag(&mut self, cx: &mut Context<Self>) {
        let Some(drag) = self.split_drag.take() else {
            return;
        };
        // Commit only while the dragged split still lives in the VISIBLE
        // active tab of the selected workspace; a tab switch mid-drag
        // cancels instead of mutating a tab the user stopped looking at.
        let target = drag.preview_ratio.and_then(|ratio| {
            let visible = self
                .selected_workspace_dump()
                .and_then(|workspace| workspace.active_tab)
                == Some(drag.tab_id);
            if !visible {
                return None;
            }
            let tab = self.active_tab_by_id(drag.tab_id)?;
            let (axis, _, _, _) = pane_split_at_path(&tab.tree, &drag.path)?;
            (axis == drag.axis).then(|| (drag.tab_id, drag.path.clone(), ratio))
        });
        if let Some((tab_id, path, ratio)) = target {
            self.dispatch(
                AppCommand::Pane(PaneCommand::ResizeSplit {
                    tab_id,
                    path,
                    ratio,
                }),
                cx,
            );
        }
        cx.notify();
    }

    fn workspace_by_id(&self, workspace_id: WorkspaceId) -> Option<&WorkspaceDump> {
        workspace_dump_for_snapshot(&self.snapshot, workspace_id)
    }

    /// Looks a workspace up in a specific connection's projection so IDs that
    /// collide across local and remote servers never mix up rows.
    fn workspace_by_id_in(
        &self,
        connection_id: ConnectionId,
        workspace_id: WorkspaceId,
    ) -> Option<&WorkspaceDump> {
        self.connection_by_id(connection_id)
            .and_then(|connection| workspace_dump_for_snapshot(&connection.snapshot, workspace_id))
    }

    fn agent_by_pane_id_in(
        &self,
        connection_id: ConnectionId,
        pane_id: PaneId,
    ) -> Option<&AgentDump> {
        self.connection_by_id(connection_id).and_then(|connection| {
            connection
                .snapshot
                .agents
                .iter()
                .find(|agent| agent.pane_id == pane_id)
        })
    }

    fn workspace_dumps(&self) -> Vec<&WorkspaceDump> {
        if self.snapshot.workspaces.is_empty() {
            self.snapshot.workspace.iter().collect()
        } else {
            self.snapshot.workspaces.iter().collect()
        }
    }

    fn agent_by_pane_id(&self, pane_id: PaneId) -> Option<&AgentDump> {
        self.snapshot
            .agents
            .iter()
            .find(|agent| agent.pane_id == pane_id)
    }

    fn pi_agent_running_for_pane(&self, pane_id: PaneId) -> bool {
        self.agent_by_pane_id(pane_id).is_some_and(|agent| {
            agent.kind == AgentKind::Pi
                && matches!(agent.status, crate::surface::TerminalStatus::Running)
        })
    }

    fn pi_agent_running_for_terminal(&self, terminal_id: TerminalId) -> bool {
        self.snapshot.agents.iter().any(|agent| {
            agent.terminal_id == terminal_id
                && agent.kind == AgentKind::Pi
                && matches!(agent.status, crate::surface::TerminalStatus::Running)
        })
    }

    fn workspace_contains_pane(&self, workspace_id: WorkspaceId, pane_id: PaneId) -> bool {
        self.workspace_by_id(workspace_id)
            .is_some_and(|workspace| workspace_active_tab_contains_pane(workspace, pane_id))
    }

    fn tab_by_id(&self, tab_id: TabId) -> Option<&TabDump> {
        self.selected_workspace_dump()?
            .tabs
            .iter()
            .find(|tab| tab.id == tab_id)
    }

    fn active_tab_by_id(&self, tab_id: TabId) -> Option<&TabDump> {
        let workspace = self.selected_workspace_dump()?;
        (workspace.active_tab == Some(tab_id))
            .then(|| workspace.tabs.iter().find(|tab| tab.id == tab_id))
            .flatten()
    }

    fn context_menu_target_exists(&self, target: ContextMenuTarget) -> bool {
        match target {
            ContextMenuTarget::Connection(connection_id) => self
                .connection_by_id(connection_id)
                .is_some_and(|connection| connection.kind == WorkspaceConnectionKind::Remote),
            ContextMenuTarget::Workspace {
                connection_id,
                workspace_id,
            } => self
                .connection_by_id(connection_id)
                .is_some_and(|connection| {
                    workspace_exists_in_snapshot(&connection.snapshot, workspace_id)
                }),
            ContextMenuTarget::Tab(tab_id) => self.tab_by_id(tab_id).is_some(),
            ContextMenuTarget::Agent {
                connection_id,
                pane_id,
            } => self
                .connection_by_id(connection_id)
                .is_some_and(|connection| {
                    connection.snapshot.agents.iter().any(|agent| {
                        agent.pane_id == pane_id
                            && matches!(agent.status, crate::surface::TerminalStatus::Running)
                    })
                }),
        }
    }

    fn rename_target_exists(&self, target: RenameTarget) -> bool {
        match target {
            RenameTarget::Tab(tab_id) => self.tab_by_id(tab_id).is_some(),
            RenameTarget::Agent {
                connection_id,
                pane_id,
            } => self
                .agent_by_pane_id_in(connection_id, pane_id)
                .is_some_and(|agent| {
                    matches!(agent.status, crate::surface::TerminalStatus::Running)
                }),
        }
    }

    fn dialog_target_exists(&self, dialog: DialogState) -> bool {
        match dialog {
            DialogState::ConfirmCloseWorkspace { workspace_id } => {
                self.workspace_by_id(workspace_id).is_some()
            }
            DialogState::ConnectRemote => true,
            DialogState::RenameWorkspace {
                connection_id,
                workspace_id,
            } => self
                .workspace_by_id_in(connection_id, workspace_id)
                .is_some(),
        }
    }

    /// True while a dialog with an editable text input is open.
    fn dialog_is_text_input(&self) -> bool {
        matches!(
            self.dialog,
            Some(DialogState::ConnectRemote) | Some(DialogState::RenameWorkspace { .. })
        )
    }

    /// The text-input dialog's confirm action is only available while the
    /// input has non-whitespace content.
    fn dialog_input_is_valid(&self) -> bool {
        self.dialog_is_text_input() && !self.dialog_input.trim().is_empty()
    }

    fn set_dialog_input(&mut self, value: String, window: &mut Window, cx: &mut Context<Self>) {
        self.dialog_input = value;
        self.dialog_caret = self.dialog_input.len();
        self.dialog_ime_base = self.dialog_caret;
        self.dialog_caret_visible = true;
        self.remote_connection_error = None;
        self.dialog_input_generation += 1;
        self.start_caret_blink(cx);
        self.focus_handle.focus(window, cx);
        cx.notify();
    }

    /// Toggles the input caret roughly twice per second while a text-input
    /// dialog stays open. Each dialog generation owns exactly one blink
    /// task: when a newer dialog opens or the view drops, the older loop
    /// notices and stops.
    fn start_caret_blink(&mut self, cx: &mut Context<Self>) {
        let generation = self.dialog_input_generation;
        cx.spawn(async move |entity, cx| {
            loop {
                cx.background_executor()
                    .spawn(CaretSleep::after(std::time::Duration::from_millis(530)))
                    .await;
                let still_owned = entity
                    .update(cx, |view, cx| {
                        if view.dialog_input_generation == generation && view.dialog_is_text_input()
                        {
                            view.dialog_caret_visible = !view.dialog_caret_visible;
                            cx.notify();
                            true
                        } else {
                            false
                        }
                    })
                    .unwrap_or(false);
                if !still_owned {
                    break;
                }
            }
        })
        .detach();
    }

    fn dispatch_close_workspace(&mut self, workspace_id: WorkspaceId, cx: &mut Context<Self>) {
        self.dispatch(
            AppCommand::Workspace(WorkspaceCommand::Close {
                workspace_id: Some(workspace_id),
            }),
            cx,
        );
    }

    fn request_close_workspace(
        &mut self,
        workspace_id: WorkspaceId,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(workspace) = self.workspace_by_id(workspace_id) else {
            return;
        };
        let has_tabs = !workspace.tabs.is_empty();
        self.context_menu = None;
        if !has_tabs {
            self.dispatch_close_workspace(workspace_id, cx);
            cx.notify();
        } else {
            self.dialog = Some(DialogState::ConfirmCloseWorkspace { workspace_id });
            self.focus_handle.focus(window, cx);
            cx.notify();
        }
    }

    fn confirm_dialog(&mut self, cx: &mut Context<Self>) {
        let Some(dialog) = self.dialog.take() else {
            return;
        };
        match dialog {
            DialogState::ConfirmCloseWorkspace { workspace_id } => {
                self.dispatch_close_workspace(workspace_id, cx);
            }
            DialogState::ConnectRemote => {
                if self.dialog_input.trim().is_empty() {
                    self.dialog = Some(DialogState::ConnectRemote);
                    return;
                }
                let destination = match crate::remote::validate_ssh_destination(&self.dialog_input)
                {
                    Ok(destination) => destination,
                    Err(error) => {
                        self.dialog = Some(DialogState::ConnectRemote);
                        self.remote_connection_error = Some(error.to_string());
                        cx.notify();
                        return;
                    }
                };
                if self.remote_connection_pending {
                    self.dialog = Some(DialogState::ConnectRemote);
                    return;
                }
                let Some(application) = self.application.clone() else {
                    self.dialog = Some(DialogState::ConnectRemote);
                    self.remote_connection_error =
                        Some("Remote connections are unavailable in this view".to_owned());
                    cx.notify();
                    return;
                };
                self.dialog = Some(DialogState::ConnectRemote);
                self.remote_connection_pending = true;
                self.remote_connection_error = None;
                cx.notify();
                application.connect_remote(destination, cx.entity().downgrade(), cx);
                return;
            }
            DialogState::RenameWorkspace {
                connection_id,
                workspace_id,
            } => {
                let title = self.dialog_input.trim().to_owned();
                if title.is_empty() {
                    self.dialog = Some(DialogState::RenameWorkspace {
                        connection_id,
                        workspace_id,
                    });
                    cx.notify();
                    return;
                }
                self.clear_dialog_input();
                self.dispatch_on(
                    connection_id,
                    AppCommand::Workspace(WorkspaceCommand::Rename {
                        workspace_id: Some(workspace_id),
                        title,
                    }),
                    cx,
                );
            }
        }
        cx.notify();
    }

    fn clear_dialog_input(&mut self) {
        self.dialog_input.clear();
        self.dialog_caret = 0;
    }

    fn cancel_dialog(&mut self, cx: &mut Context<Self>) {
        if self.dialog.take().is_some() {
            self.clear_dialog_input();
            self.remote_connection_error = None;
            self.remote_connection_pending = false;
            cx.notify();
        }
    }

    fn begin_connect_remote(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.has_transient_ui() {
            return;
        }
        self.context_menu = None;
        self.clear_dialog_input();
        self.remote_connection_error = None;
        self.remote_connection_pending = false;
        self.dialog = Some(DialogState::ConnectRemote);
        self.set_dialog_input(String::new(), window, cx);
    }

    pub(crate) fn finish_remote_connection(
        &mut self,
        result: Result<ConnectionId, String>,
        cx: &mut Context<Self>,
    ) {
        if !self.remote_connection_pending || self.dialog != Some(DialogState::ConnectRemote) {
            if let Ok(connection_id) = result
                && let Some(application) = self.application.clone()
            {
                application.disconnect_connection(connection_id, cx);
            }
            return;
        }
        self.remote_connection_pending = false;
        match result {
            Ok(connection_id) => {
                self.dialog = None;
                self.clear_dialog_input();
                self.remote_connection_error = None;
                self.select_connection_locally(connection_id, cx);
            }
            Err(error) => self.remote_connection_error = Some(error),
        }
        cx.notify();
    }

    fn handle_dialog_key(&mut self, event: &KeyDownEvent, cx: &mut Context<Self>) -> bool {
        if self.dialog.is_none() {
            return false;
        }
        if self.remote_connection_pending && self.dialog == Some(DialogState::ConnectRemote) {
            if event.keystroke.key == "escape" {
                self.cancel_dialog(cx);
            }
            return true;
        }
        let key = event.keystroke.key.as_str();
        if self.dialog_is_text_input() {
            match key {
                "enter" | "return" => self.confirm_dialog(cx),
                "escape" => self.cancel_dialog(cx),
                "backspace" => {
                    self.edit_dialog_input_before_caret(cx);
                }
                "delete" => {
                    self.edit_dialog_input_after_caret(cx);
                }
                "left" => self.move_dialog_caret(-1, cx),
                "right" => self.move_dialog_caret(1, cx),
                "home" => self.set_dialog_caret(0, cx),
                "end" => self.set_dialog_caret(self.dialog_input.len(), cx),
                _ if !event.keystroke.modifiers.platform
                    && !event.keystroke.modifiers.control
                    && !event.keystroke.modifiers.alt
                    && event.keystroke.key_char.is_some() =>
                {
                    if let Some(character) = event.keystroke.key_char.as_deref() {
                        let insert_at = self.dialog_caret.clamp(0, self.dialog_input.len());
                        self.dialog_input.insert_str(insert_at, character);
                        self.dialog_caret = insert_at + character.len();
                        self.dialog_ime_base = self.dialog_caret;
                        self.dialog_caret_visible = true;
                        self.remote_connection_error = None;
                        cx.notify();
                    }
                }
                _ => {}
            }
            return true;
        }
        match (self.dialog, key) {
            (_, "enter" | "return") => self.confirm_dialog(cx),
            (_, "escape") => self.cancel_dialog(cx),
            _ => {}
        }
        true
    }

    fn set_dialog_caret(&mut self, byte_offset: usize, cx: &mut Context<Self>) {
        let byte_offset = byte_offset.clamp(0, self.dialog_input.len());
        if self.dialog_caret == byte_offset {
            return;
        }
        self.dialog_caret = byte_offset;
        self.dialog_ime_base = byte_offset;
        self.dialog_caret_visible = true;
        cx.notify();
    }

    /// Moves the caret by `delta` whole characters, clamped to the value.
    fn move_dialog_caret(&mut self, delta: isize, cx: &mut Context<Self>) {
        if self.dialog_input.is_empty() || delta == 0 {
            return;
        }
        let mut boundaries: Vec<usize> = self
            .dialog_input
            .char_indices()
            .map(|(index, _)| index)
            .collect();
        boundaries.push(self.dialog_input.len());
        let rank = boundaries
            .iter()
            .rposition(|boundary| *boundary <= self.dialog_caret)
            .unwrap_or(0);
        let target = if delta < 0 {
            rank.saturating_sub((-delta) as usize)
        } else {
            (rank + delta as usize).min(boundaries.len() - 1)
        };
        self.set_dialog_caret(boundaries[target], cx);
    }

    fn edit_dialog_input_before_caret(&mut self, cx: &mut Context<Self>) {
        if self.dialog_caret == 0 {
            return;
        }
        let char_start = self
            .dialog_input
            .char_indices()
            .rev()
            .find(|(index, _)| *index < self.dialog_caret)
            .map(|(index, _)| index)
            .unwrap_or(0);
        self.dialog_input
            .replace_range(char_start..self.dialog_caret, "");
        self.dialog_caret = char_start;
        self.dialog_ime_base = char_start;
        self.dialog_caret_visible = true;
        self.remote_connection_error = None;
        cx.notify();
    }

    fn edit_dialog_input_after_caret(&mut self, cx: &mut Context<Self>) {
        let caret = self.dialog_caret.clamp(0, self.dialog_input.len());
        let Some((char_start, character)) = self
            .dialog_input
            .char_indices()
            .find(|(index, _)| *index >= caret)
        else {
            return;
        };
        let char_end = char_start + character.len_utf8();
        self.dialog_input.replace_range(char_start..char_end, "");
        self.dialog_ime_base = caret;
        self.dialog_caret_visible = true;
        self.remote_connection_error = None;
        cx.notify();
    }

    fn begin_rename_active_workspace(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if let Some(workspace_id) = self.active_workspace_id() {
            self.begin_rename_workspace(self.active_connection, workspace_id, window, cx);
        }
    }

    fn begin_rename_active_tab(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let tab_id = self
            .active_workspace_id()
            .and_then(|workspace_id| self.workspace_by_id(workspace_id))
            .and_then(|workspace| workspace.active_tab);
        if let Some(tab_id) = tab_id {
            self.begin_rename_tab(tab_id, window, cx);
        }
    }

    /// Opens the shared text-input dialog for renaming a workspace. The
    /// dialog is scoped to the owning connection, so workspaces whose IDs
    /// collide across local and remote servers cannot be mixed up.
    fn begin_rename_workspace(
        &mut self,
        connection_id: ConnectionId,
        workspace_id: WorkspaceId,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if self.has_transient_ui() {
            return;
        }
        let Some(workspace_title) = self
            .workspace_by_id_in(connection_id, workspace_id)
            .map(|workspace| workspace.title.clone())
        else {
            return;
        };
        self.context_menu = None;
        self.rename_target = None;
        self.rename_value.clear();
        self.dialog = Some(DialogState::RenameWorkspace {
            connection_id,
            workspace_id,
        });
        self.set_dialog_input(workspace_title, window, cx);
    }

    fn begin_rename_tab(&mut self, tab_id: TabId, window: &mut Window, cx: &mut Context<Self>) {
        if self.has_transient_ui() {
            return;
        }
        let Some(tab_title) = self
            .selected_workspace_dump()
            .and_then(|workspace| workspace.tabs.iter().find(|tab| tab.id == tab_id))
            .map(|tab| tab.title.clone())
        else {
            return;
        };
        self.context_menu = None;
        self.rename_target = Some(RenameTarget::Tab(tab_id));
        self.rename_value = tab_title;
        self.focus_handle.focus(window, cx);
        cx.notify();
    }

    fn begin_rename_agent(
        &mut self,
        connection_id: ConnectionId,
        pane_id: PaneId,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if self.has_transient_ui() {
            return;
        }
        let Some(agent_label) = self
            .agent_by_pane_id_in(connection_id, pane_id)
            .filter(|agent| matches!(agent.status, crate::surface::TerminalStatus::Running))
            .map(|agent| agent.display_label().to_owned())
        else {
            return;
        };
        self.context_menu = None;
        self.rename_target = Some(RenameTarget::Agent {
            connection_id,
            pane_id,
        });
        self.rename_value = agent_label;
        self.focus_handle.focus(window, cx);
        cx.notify();
    }

    fn cancel_rename(&mut self, cx: &mut Context<Self>) {
        if self.rename_target.take().is_some() {
            self.rename_value.clear();
            cx.notify();
        }
    }

    fn commit_rename(&mut self, cx: &mut Context<Self>) {
        let Some(target) = self.rename_target.take() else {
            return;
        };
        let title = self.rename_value.trim().to_owned();
        self.rename_value.clear();
        if title.is_empty() && !matches!(target, RenameTarget::Agent { .. }) {
            cx.notify();
            return;
        }
        match target {
            RenameTarget::Tab(tab_id) => {
                self.dispatch(
                    AppCommand::Tab(TabCommand::Rename {
                        tab_id: Some(tab_id),
                        title,
                    }),
                    cx,
                );
            }
            RenameTarget::Agent {
                connection_id,
                pane_id,
            } => {
                self.dispatch_on(
                    connection_id,
                    AppCommand::Pane(PaneCommand::RenameAgent {
                        pane_id: Some(pane_id),
                        label: title,
                    }),
                    cx,
                );
            }
        }
        cx.notify();
    }

    fn handle_rename_key(&mut self, event: &KeyDownEvent, cx: &mut Context<Self>) -> bool {
        if self.rename_target.is_none() {
            return false;
        }
        let key = event.keystroke.key.as_str();
        match key {
            "enter" | "return" => self.commit_rename(cx),
            "escape" => self.cancel_rename(cx),
            "backspace" => {
                self.rename_value.pop();
                cx.notify();
            }
            _ if !event.keystroke.modifiers.platform
                && !event.keystroke.modifiers.control
                && !event.keystroke.modifiers.alt
                && event.keystroke.key_char.is_some() =>
            {
                if let Some(character) = event.keystroke.key_char.as_deref() {
                    self.rename_value.push_str(character);
                    cx.notify();
                }
            }
            _ => {}
        }
        true
    }

    fn measured_terminal_metrics(&self, window: &Window) -> TerminalMetrics {
        let font_size = px(self.config.terminal.font_size);
        let font = font(self.config.terminal.font_family.clone());
        let text_system = window.text_system();
        let cell_width = text_system
            .ch_advance(text_system.resolve_font(&font), font_size)
            .ok()
            .map(f32::from)
            .filter(|width| *width > 0.0)
            .unwrap_or(DEFAULT_TERMINAL_CELL_WIDTH);
        TerminalMetrics {
            cell_width,
            line_height: self.config.terminal.line_height,
            scale_factor: window.scale_factor(),
        }
    }

    /// Routes a terminal command through the transport of the connection
    /// that owns the terminal. The window's `self.client` is the
    /// active connection's transport, so sending by terminal ID through it
    /// after a connection switch would reach the wrong server (local vs
    /// remote) and be dropped as an unknown terminal.
    fn enqueue_terminal_command(&self, terminal_id: TerminalId, command: TerminalCommand) -> bool {
        let client = self
            .connection_by_id(self.active_connection)
            .map(|connection| connection.client.clone())
            .unwrap_or_else(|| self.client.clone());
        match client.enqueue(AppCommand::Terminal(command)) {
            Ok(()) => true,
            Err(error) => {
                tracing::warn!(
                    target: "water::ui",
                    terminal_id = %terminal_id,
                    ?error,
                    "could not enqueue terminal command"
                );
                false
            }
        }
    }

    /// Viewport moves are GUI-local in the raw-stream architecture. They are
    /// queued to the emulator worker; the visual accumulator covers the short
    /// delay until its next snapshot acknowledges the new viewport.
    ///
    /// When the viewport moves away from the bottom, the emulator's pin flag
    /// switches to a semantic user coordinate. Alacritty may advance its
    /// physical `display_offset` as output grows, but the rendered rows stay
    /// anchored to the user's reading position. Scrolling back to the bottom
    /// clears the pin via `ScrollTo(0)` / `scroll_to_bottom`.
    fn apply_local_viewport(
        &mut self,
        terminal_id: TerminalId,
        target: Option<i64>,
        delta: Option<i64>,
        _cx: &mut Context<Self>,
    ) {
        let Some(application) = self.application.clone() else {
            return;
        };
        let connection_id = self.active_connection;
        if let Some(target) = target {
            // Pin when scrolling to a non-zero target; unpinned at 0.
            application.terminal_set_viewport_pinned(connection_id, terminal_id, target != 0);
            application.terminal_scroll_to(connection_id, terminal_id, target);
        }
        if let Some(delta) = delta {
            // For delta scrolls, compute the expected target from the
            // current snapshot to decide whether to pin.
            let current = self
                .terminal_snapshot_for(terminal_id)
                .map(|s| s.viewport_position)
                .unwrap_or(0);
            let expected = (current + delta).max(0);
            if expected != 0 {
                application.terminal_set_viewport_pinned(connection_id, terminal_id, true);
            }
            application.terminal_scroll_by(connection_id, terminal_id, delta);
        }
    }

    /// Returns a terminal to its live viewport before user input reaches the
    /// shell. Cancel every UI-local scroll source first so a request already
    /// scheduled for the next frame cannot pull the viewport back into
    /// history after the input-triggered jump.
    fn focus_terminal_live_bottom(&mut self, terminal_id: TerminalId, cx: &mut Context<Self>) {
        self.selection = None;
        self.selection_autoscroll = None;
        self.active_trackpad_scrolls.remove(&terminal_id);
        self.mouse_scroll_animations.remove(&terminal_id);
        let had_pending_request = self
            .pending_viewport_requests
            .remove(&terminal_id)
            .is_some();

        let viewport_position = self
            .terminal_snapshot_for(terminal_id)
            .map(|snapshot| snapshot.viewport_position)
            .unwrap_or_default();
        let state = self
            .scroll_accumulators
            .entry(terminal_id)
            .or_insert_with(|| TerminalScrollState::new(viewport_position));
        let was_scrolled = had_pending_request
            || state.observed_viewport_position != 0
            || state.visual_unacked_rows != 0.0
            || state.requested_viewport_position != 0;
        if !was_scrolled {
            return;
        }

        // Preserve the current snapshot as the reconciliation base while
        // painting as close to the live bottom as its prepared rows allow.
        // When the emulator acknowledges ScrollTo(0), reconciliation reduces
        // this visual offset back to zero without a backwards flash.
        state.visual_unacked_rows = -(state.observed_viewport_position as f32);
        state.requested_viewport_position = 0;
        state.request_started_at = Some(Instant::now());
        scroll_stat_inc(&SCROLL_VIEWPORT_REQUESTS);
        scroll_stat_max_unacked(state.visual_unacked_rows);

        self.apply_local_viewport(terminal_id, Some(0), None, cx);
        cx.notify();
    }

    /// Like [`Self::focus_terminal_live_bottom`] but for output-driven jumps
    /// while Pi's regular TUI is running and the user is browsing scrollback:
    /// the live-bottom request is issued (so returning to the bottom is
    /// immediate once the user asks for it) but the visual unacked offset is
    /// not pre-painted, so the current reading position stays on screen until
    /// the user explicitly returns to the bottom.
    fn focus_terminal_live_bottom_guarded(&mut self, terminal_id: TerminalId, cx: &mut Context<Self>) {
        self.selection = None;
        self.selection_autoscroll = None;
        self.active_trackpad_scrolls.remove(&terminal_id);
        self.mouse_scroll_animations.remove(&terminal_id);
        self.pending_viewport_requests.remove(&terminal_id);

        let viewport_position = self
            .terminal_snapshot_for(terminal_id)
            .map(|snapshot| snapshot.viewport_position)
            .unwrap_or_default();
        let state = self
            .scroll_accumulators
            .entry(terminal_id)
            .or_insert_with(|| TerminalScrollState::new(viewport_position));
        if viewport_position == 0 {
            return;
        }
        // Rebase without a visual jump: the observed viewport stays where the
        // user is reading; the requested target is the live bottom.
        state.visual_unacked_rows = 0.0;
        state.requested_viewport_position = 0;
        state.request_started_at = Some(Instant::now());
        self.apply_local_viewport(terminal_id, Some(0), None, cx);
        cx.notify();
    }

    /// Refreshes the locally rendered snapshots for terminals whose raw
    /// stream advanced, then requests one repaint for the whole view.
    pub(crate) fn apply_terminal_events(&mut self, changed: &[TerminalId], cx: &mut Context<Self>) {
        if changed.is_empty() {
            return;
        }
        let Some(application) = self.application.clone() else {
            return;
        };
        let connection_id = self.active_connection;
        let mut displayed = BTreeSet::new();
        if let Some(workspace) = self.selected_workspace_dump()
            && let Some(active_tab) = workspace.active_tab
            && let Some(tab) = workspace.tabs.iter().find(|tab| tab.id == active_tab)
        {
            collect_terminal_ids(&tab.tree, &mut displayed);
        }
        let mut needs_notify = false;
        for &terminal_id in changed {
            if !displayed.contains(&terminal_id) {
                // The connection-level emulator was still advanced above;
                // avoid materializing rows or repainting a hidden tab.
                self.terminal_snapshots.remove(&terminal_id);
                crate::metrics::inc(crate::metrics::hidden_terminal_updates());
                continue;
            }
            // While the user is browsing scrollback (viewport above the
            // live bottom), output must not pull the viewport to the live
            // bottom. The pinned emulator viewport plus the disabled
            // smooth-scroll paint offset keep the reading position stable
            // while new output grows the grid behind it. Keystrokes
            // (focus_terminal_live_bottom in the input handler) return the
            // viewport to the live bottom, which also trims the scrollback
            // back to the configured limit.
            let is_browsing = self
                .terminal_snapshot_for(terminal_id)
                .is_some_and(|snapshot| snapshot.viewport_position > 0);
            if is_browsing {
                self.focus_terminal_live_bottom_guarded(terminal_id, cx);
            }
            // At the live bottom (viewport_position == 0) the emulator is
            // already in live-follow mode — no action needed on output.
            let previous = self.terminal_snapshots.get(&terminal_id).cloned();
            let previous_ref = previous.as_deref();
            let snapshot = application.terminal_snapshot(connection_id, terminal_id, previous_ref);
            match (previous, snapshot) {
                (Some(previous), Some(snapshot)) => {
                    reconcile_local_viewport_snapshot(
                        &mut self.selection,
                        &mut self.scroll_accumulators,
                        terminal_id,
                        &previous,
                        &snapshot,
                    );
                    self.terminal_snapshots.insert(terminal_id, snapshot);
                }
                (None, Some(snapshot)) => {
                    self.terminal_snapshots.insert(terminal_id, snapshot);
                }
                (_, None) => {
                    self.terminal_snapshots.remove(&terminal_id);
                    if let Some(selection) = &self.selection
                        && selection.terminal_id == terminal_id
                    {
                        self.selection = None;
                    }
                }
            }
            needs_notify = true;
        }
        if needs_notify {
            crate::metrics::inc(crate::metrics::terminal_notifies());
            cx.notify();
        }
    }

    fn active_terminal_id(&self) -> Option<TerminalId> {
        let focused_pane = self.focused_pane?;
        let workspace = self.selected_workspace_dump()?;
        let active_tab_id = workspace.active_tab?;
        let tab = workspace.tabs.iter().find(|tab| tab.id == active_tab_id)?;
        terminal_id_for_pane(&tab.tree, focused_pane)
    }

    fn active_terminal_snapshot(&self) -> Option<&TerminalSnapshot> {
        let focused_pane = self.focused_pane?;
        let workspace_id = self.selected_workspace?;
        let workspace = workspace_dump_for_snapshot(&self.snapshot, workspace_id)?;
        let active_tab_id = workspace.active_tab?;
        let tab = workspace.tabs.iter().find(|tab| tab.id == active_tab_id)?;
        let terminal_id = terminal_id_for_pane(&tab.tree, focused_pane)?;
        self.terminal_snapshots
            .get(&terminal_id)
            .map(std::sync::Arc::as_ref)
    }

    fn ime_marked_text_for(&self, terminal_id: TerminalId) -> Option<String> {
        (self.ime_terminal == Some(terminal_id) && !self.ime_marked_text.is_empty())
            .then(|| self.ime_marked_text.clone())
    }

    fn clear_ime(&mut self) {
        self.ime_terminal = None;
        self.ime_marked_text.clear();
        self.ime_selected_range = 0..0;
    }

    fn reset_ime_marked_text(&mut self) {
        self.ime_marked_text.clear();
        self.ime_selected_range = 0..0;
    }

    fn terminal_id_for_pane(&self, pane_id: PaneId) -> Option<TerminalId> {
        let workspace = self.selected_workspace_dump()?;
        workspace
            .tabs
            .iter()
            .find_map(|tab| terminal_id_for_pane(&tab.tree, pane_id))
    }

    /// Pane IDs are not unique across connections: the local and every
    /// remote server allocate IDs from their own counters, so a pane ID
    /// must never be resolved without its owning connection. This lookup is
    /// scoped to the window's active connection.
    fn terminal_id_for_pane_in_active_connection(&self, pane_id: PaneId) -> Option<TerminalId> {
        let snapshot = self
            .connection_by_id(self.active_connection)?
            .snapshot
            .clone();
        let workspaces = if snapshot.workspaces.is_empty() {
            snapshot.workspace.into_iter().collect::<Vec<_>>()
        } else {
            snapshot.workspaces
        };
        workspaces.iter().find_map(|workspace| {
            workspace
                .tabs
                .iter()
                .find_map(|tab| terminal_id_for_pane(&tab.tree, pane_id))
        })
    }

    fn terminal_bounds_for(&self, terminal_id: TerminalId) -> Option<Bounds<gpui::Pixels>> {
        self.terminal_bounds
            .lock()
            .expect("terminal bounds poisoned")
            .get(&terminal_id)
            .copied()
    }

    fn terminal_snapshot_for(&self, terminal_id: TerminalId) -> Option<&TerminalSnapshot> {
        self.terminal_snapshots
            .get(&terminal_id)
            .map(std::sync::Arc::as_ref)
    }

    /// Ensures the raw stream is attached and returns the locally rendered
    /// snapshot for a terminal (building it on first use).
    fn local_terminal_snapshot(
        &mut self,
        terminal_id: TerminalId,
        _cx: &mut Context<Self>,
    ) -> Option<std::sync::Arc<TerminalSnapshot>> {
        if let Some(cached) = self.terminal_snapshots.get(&terminal_id).cloned() {
            return Some(cached);
        }
        let Some(application) = self.application.clone() else {
            return None;
        };
        let connection_id = self.active_connection;
        application.ensure_terminal_attached(connection_id, terminal_id);
        let snapshot = application.terminal_snapshot(connection_id, terminal_id, None)?;
        self.terminal_snapshots
            .insert(terminal_id, snapshot.clone());
        Some(snapshot)
    }

    fn terminal_scroll_offset_for_snapshot(&self, snapshot: &TerminalSnapshot) -> f32 {
        let offset = self
            .scroll_accumulators
            .get(&snapshot.terminal_id)
            .map(|state| state.visual_unacked_rows)
            .unwrap_or(0.0);
        // The painted content is limited to the materialized overscan window
        // (rows_before/rows_after). The user can still *scroll* to the full
        // history (the viewport_position moves via the local emulator), but
        // the unacked visual offset that shifts the paint is capped at the
        // overscan so the paint never addresses rows the snapshot cannot
        // resolve. Once the viewport ACK lands, the snapshot is rebuilt with
        // a new overscan window centered on the new viewport.
        let overscan = snapshot.rows_before.len() as f32;
        // Downward (negative) movement is additionally bounded by the
        // distance to the materialized live bottom (last_source_row).
        // Without this clamp, an input-triggered jump to the live bottom
        // (visual_unacked_rows = -viewport_position) is painted clamped to
        // -rows_after.len(), shifting the window up for the frames before
        // the ScrollTo(0) ACK lands.
        let live_bottom_limit = snapshot.last_source_row.min(overscan as i64) as f32;
        offset.clamp(-live_bottom_limit, overscan)
    }

    fn accumulate_terminal_scroll(
        &mut self,
        terminal_id: TerminalId,
        delta_rows: f32,
    ) -> (Option<i64>, bool) {
        let (viewport_position, at_history_start, at_live_bottom) = self
            .terminal_snapshot_for(terminal_id)
            .map_or((0, false, false), |snapshot| {
                (
                    snapshot.viewport_position,
                    snapshot.rows_before.is_empty(),
                    snapshot.rows_after.is_empty(),
                )
            });
        self.scroll_accumulators
            .entry(terminal_id)
            .or_insert_with(|| TerminalScrollState::new(viewport_position))
            .accumulate_with_boundaries(delta_rows, at_history_start, at_live_bottom)
    }

    fn queue_terminal_viewport_request(
        &mut self,
        terminal_id: TerminalId,
        pane_id: PaneId,
        target: i64,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        // Input can arrive much faster than a display refresh. Preserve every
        // fractional delta and commit the latest local target once per frame.
        record_latest_viewport_request(
            &mut self.pending_viewport_requests,
            terminal_id,
            pane_id,
            target,
        );
        if self.viewport_request_frame_pending {
            return;
        }
        self.viewport_request_frame_pending = true;
        cx.on_next_frame(window, |this, _window, _cx| {
            this.viewport_request_frame_pending = false;
            for (terminal_id, (_pane_id, target)) in
                std::mem::take(&mut this.pending_viewport_requests)
            {
                this.apply_local_viewport(terminal_id, Some(target), None, _cx);
            }
        });
    }

    fn visual_terminal_position(&self, terminal_id: TerminalId) -> f32 {
        let position = self.scroll_accumulators.get(&terminal_id).map_or_else(
            || {
                self.terminal_snapshot_for(terminal_id)
                    .map_or(0.0, |snapshot| snapshot.viewport_position as f32)
            },
            |state| state.observed_viewport_position as f32 + state.visual_unacked_rows,
        );
        self.clamp_terminal_visual_position(terminal_id, position)
    }

    fn clamp_terminal_visual_position(&self, terminal_id: TerminalId, mut position: f32) -> f32 {
        let Some(snapshot) = self.terminal_snapshot_for(terminal_id) else {
            return position;
        };
        let base = snapshot.viewport_position as f32;
        if snapshot.history_len == 0 {
            position = position.min(base);
        }
        if snapshot.rows_after.is_empty() {
            position = position.max(base);
        }
        position
    }

    fn begin_mouse_scroll_animation(
        &mut self,
        terminal_id: TerminalId,
        pane_id: PaneId,
        delta_rows: f32,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> bool {
        let current = self.visual_terminal_position(terminal_id);
        let carried_target = self
            .mouse_scroll_animations
            .get(&terminal_id)
            .map_or(current, |animation| animation.target_position);
        let carried_target = self.clamp_terminal_visual_position(terminal_id, carried_target);
        let target_position =
            self.clamp_terminal_visual_position(terminal_id, carried_target + delta_rows);
        if (target_position - current).abs() < f32::EPSILON {
            self.mouse_scroll_animations.remove(&terminal_id);
            return false;
        }

        // Move on the input event itself so a fresh notch never waits for the
        // first animation callback. The remaining distance is display-paced.
        let immediate_delta = (target_position - current) * MOUSE_SCROLL_IMMEDIATE_FRACTION;
        let (request, repaint) = self.accumulate_terminal_scroll(terminal_id, immediate_delta);
        if let Some(target) = request {
            self.queue_terminal_viewport_request(terminal_id, pane_id, target, window, cx);
        }
        let start_position = self.visual_terminal_position(terminal_id);
        self.mouse_scroll_animations.insert(
            terminal_id,
            TerminalMouseScrollAnimation {
                pane_id,
                start_position,
                target_position,
                started_at: Instant::now(),
            },
        );
        self.request_mouse_scroll_frame(window, cx);
        repaint
    }

    fn request_mouse_scroll_frame(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.mouse_scroll_frame_pending {
            return;
        }
        self.mouse_scroll_frame_pending = true;
        cx.on_next_frame(window, |this, window, cx| {
            this.mouse_scroll_frame_pending = false;
            let now = Instant::now();
            let terminal_ids = this
                .mouse_scroll_animations
                .keys()
                .copied()
                .collect::<Vec<_>>();
            let mut finished = Vec::new();
            let mut repaint = false;
            for terminal_id in terminal_ids {
                let Some(animation) = this.mouse_scroll_animations.get(&terminal_id).cloned()
                else {
                    continue;
                };
                let (position, done) = animation.position_at(now);
                let delta_rows = position - this.visual_terminal_position(terminal_id);
                let (request, changed) = this.accumulate_terminal_scroll(terminal_id, delta_rows);
                repaint |= changed;
                if let Some(target) = request {
                    this.queue_terminal_viewport_request(
                        terminal_id,
                        animation.pane_id,
                        target,
                        window,
                        cx,
                    );
                }
                if done {
                    finished.push(terminal_id);
                }
            }
            for terminal_id in finished {
                this.mouse_scroll_animations.remove(&terminal_id);
            }
            if repaint {
                cx.notify();
            }
            if !this.mouse_scroll_animations.is_empty() {
                this.request_mouse_scroll_frame(window, cx);
            }
        });
    }

    fn terminal_selection_endpoint_at(
        &self,
        terminal_id: TerminalId,
        position: Point<gpui::Pixels>,
    ) -> Option<TerminalSelectionEndpoint> {
        let snapshot = self.terminal_snapshot_for(terminal_id)?;
        let bounds = self.terminal_bounds_for(terminal_id);
        let mouse = terminal_mouse_position(position, bounds, self.terminal_metrics);
        let column = mouse
            .column
            .saturating_sub(1)
            .min(snapshot.size.columns.saturating_sub(1));
        let side = if mouse.row > snapshot.size.lines || mouse.column > snapshot.size.columns {
            TerminalSelectionSide::Right
        } else {
            mouse.side
        };
        // The painted content is shifted by the unacked scroll offset (positive
        // = scrolled up into history), so a pixel row addresses the source row
        // that the paint actually shows — not the viewport's first row.
        let scroll_offset_rows = self.terminal_scroll_offset_for_snapshot(snapshot);
        let first_available = -(snapshot.rows_before.len() as i32);
        let last_available = snapshot
            .size
            .lines
            .saturating_add(snapshot.rows_after.len())
            .saturating_sub(1) as i32;
        let row = terminal_source_row_at(
            position.y,
            bounds,
            self.terminal_metrics,
            snapshot.size.lines,
            scroll_offset_rows,
        )
        .max(first_available)
        .min(last_available);
        Some(TerminalSelectionEndpoint {
            position: TerminalCellPosition { row, column },
            side,
        })
    }

    fn terminal_cell_at(
        &self,
        terminal_id: TerminalId,
        position: Point<gpui::Pixels>,
    ) -> Option<TerminalCellPosition> {
        self.terminal_selection_endpoint_at(terminal_id, position)
            .map(|endpoint| endpoint.position)
    }

    fn begin_reported_mouse(
        &mut self,
        terminal_id: TerminalId,
        event: &MouseDownEvent,
        cx: &mut Context<Self>,
    ) -> bool {
        let Some(snapshot) = self.terminal_snapshot_for(terminal_id) else {
            return false;
        };
        if !self.config.features.mouse_reporting || !snapshot.modes.mouse_reporting {
            return false;
        }
        let mouse = TerminalMouseContext {
            modes: snapshot.modes,
            bounds: self.terminal_bounds_for(terminal_id),
            metrics: self.terminal_metrics,
        };
        let Some(text) = terminal_mouse_button_input(
            event.position,
            event.button,
            true,
            false,
            event.modifiers,
            mouse,
        ) else {
            return false;
        };
        self.reset_ime_marked_text();
        self.enqueue_terminal_command(
            terminal_id,
            TerminalCommand::SendBytes {
                terminal_id: Some(terminal_id),
                pane_id: None,
                bytes: text,
            },
        );
        self.reported_mouse = Some((terminal_id, event.button));
        self.last_reported_mouse_cell = self
            .terminal_cell_at(terminal_id, event.position)
            .map(|point| (terminal_id, point));
        cx.notify();
        true
    }

    fn begin_terminal_selection(
        &mut self,
        terminal_id: TerminalId,
        position: Point<gpui::Pixels>,
        shift_held: bool,
        cx: &mut Context<Self>,
    ) {
        if !self.config.features.selection {
            return;
        }
        let Some(endpoint) = self.terminal_selection_endpoint_at(terminal_id, position) else {
            return;
        };
        self.clear_ime();
        self.selection = Some(terminal_selection_after_click(
            self.selection,
            terminal_id,
            endpoint,
            shift_held,
        ));
        self.selection_autoscroll = None;
        self.dragging_terminal = Some(terminal_id);
        cx.notify();
    }

    fn terminal_selection_autoscroll_delta(
        &self,
        terminal_id: TerminalId,
        position: Point<gpui::Pixels>,
    ) -> Option<i64> {
        if self.pi_agent_running_for_terminal(terminal_id) {
            // A running Pi regular TUI owns the primary screen. Selection
            // edge scrolling must not reveal the host scrollback until the
            // foreground process returns to the shell.
            return None;
        }
        let snapshot = self.terminal_snapshot_for(terminal_id)?;
        let bounds = self.terminal_bounds_for(terminal_id)?;
        let pane_height = f32::from(bounds.size.height);
        if pane_height <= 0.0 {
            return None;
        }
        let y_in_pane = f32::from(position.y) - f32::from(bounds.origin.y);
        let delta = terminal_selection_autoscroll_direction(y_in_pane, pane_height)?;
        let at_boundary = if delta > 0 {
            snapshot.rows_before.is_empty()
        } else {
            snapshot.rows_after.is_empty()
        };
        (!at_boundary).then_some(delta)
    }

    fn update_terminal_selection_head_at(
        &mut self,
        terminal_id: TerminalId,
        position: Point<gpui::Pixels>,
        cx: &mut Context<Self>,
    ) {
        let Some(endpoint) = self.terminal_selection_endpoint_at(terminal_id, position) else {
            return;
        };
        if let Some(selection) = self
            .selection
            .as_mut()
            .filter(|selection| selection.terminal_id == terminal_id)
            && selection.head != endpoint
        {
            selection.head = endpoint;
            cx.notify();
        }
    }

    fn request_selection_autoscroll_frame(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.selection_autoscroll_frame_pending {
            return;
        }
        self.selection_autoscroll_frame_pending = true;
        cx.on_next_frame(window, |this, window, cx| {
            this.selection_autoscroll_frame_pending = false;
            let Some(autoscroll) = this.selection_autoscroll else {
                return;
            };
            if this.dragging_terminal != Some(autoscroll.terminal_id) {
                this.selection_autoscroll = None;
                return;
            }
            let Some(delta) = this
                .terminal_selection_autoscroll_delta(autoscroll.terminal_id, autoscroll.position)
            else {
                this.selection_autoscroll = None;
                return;
            };
            this.apply_local_viewport(autoscroll.terminal_id, None, Some(delta), cx);
            this.update_terminal_selection_head_at(autoscroll.terminal_id, autoscroll.position, cx);
            this.request_selection_autoscroll_frame(window, cx);
        });
    }

    fn update_terminal_selection(
        &mut self,
        event: &MouseMoveEvent,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if self.reported_mouse.is_some() {
            self.selection_autoscroll = None;
            self.update_reported_mouse(event, cx);
            return;
        }
        let Some(terminal_id) = self.dragging_terminal else {
            return;
        };
        if event.pressed_button != Some(MouseButton::Left) {
            self.selection_autoscroll = None;
            return;
        }
        if let Some(delta) = self.terminal_selection_autoscroll_delta(terminal_id, event.position) {
            self.selection_autoscroll = Some(TerminalSelectionAutoscroll {
                terminal_id,
                position: event.position,
            });
            // Give the first edge move immediate feedback, then continue at
            // display cadence while the pointer remains in the edge band.
            self.apply_local_viewport(terminal_id, None, Some(delta), cx);
            self.request_selection_autoscroll_frame(window, cx);
        } else {
            self.selection_autoscroll = None;
        }
        self.update_terminal_selection_head_at(terminal_id, event.position, cx);
    }

    fn update_reported_mouse(&mut self, event: &MouseMoveEvent, cx: &mut Context<Self>) {
        let Some((terminal_id, button)) = self.reported_mouse else {
            return;
        };
        if event.pressed_button != Some(button) {
            return;
        }
        let Some(snapshot) = self.terminal_snapshot_for(terminal_id) else {
            return;
        };
        if !snapshot.modes.mouse_motion && !snapshot.modes.mouse_drag {
            return;
        }
        let Some(point) = self.terminal_cell_at(terminal_id, event.position) else {
            return;
        };
        if self.last_reported_mouse_cell == Some((terminal_id, point)) {
            return;
        }
        let mouse = TerminalMouseContext {
            modes: snapshot.modes,
            bounds: self.terminal_bounds_for(terminal_id),
            metrics: self.terminal_metrics,
        };
        let Some(text) =
            terminal_mouse_button_input(event.position, button, true, true, event.modifiers, mouse)
        else {
            return;
        };
        self.enqueue_terminal_command(
            terminal_id,
            TerminalCommand::SendBytes {
                terminal_id: Some(terminal_id),
                pane_id: None,
                bytes: text,
            },
        );
        self.last_reported_mouse_cell = Some((terminal_id, point));
        cx.stop_propagation();
    }

    fn finish_terminal_selection(&mut self, event: &MouseUpEvent, cx: &mut Context<Self>) {
        self.selection_autoscroll = None;
        if let Some((terminal_id, button)) = self.reported_mouse.take() {
            if button == event.button
                && let Some(snapshot) = self.terminal_snapshot_for(terminal_id)
                && let Some(text) = terminal_mouse_button_input(
                    event.position,
                    button,
                    false,
                    false,
                    event.modifiers,
                    TerminalMouseContext {
                        modes: snapshot.modes,
                        bounds: self.terminal_bounds_for(terminal_id),
                        metrics: self.terminal_metrics,
                    },
                )
            {
                self.enqueue_terminal_command(
                    terminal_id,
                    TerminalCommand::SendBytes {
                        terminal_id: Some(terminal_id),
                        pane_id: None,
                        bytes: text,
                    },
                );
            }
            self.last_reported_mouse_cell = None;
            cx.stop_propagation();
        } else if event.button == MouseButton::Left
            && self.config.features.selection
            && let Some(terminal_id) = self.dragging_terminal
            && let Some(endpoint) = self.terminal_selection_endpoint_at(terminal_id, event.position)
            && let Some(selection) = self.selection.as_mut()
            && selection.terminal_id == terminal_id
            && selection.head != endpoint
        {
            // Mouse-up is the authoritative endpoint. A platform may omit a
            // final move event, so do not leave the copied range one cell
            // behind the pointer.
            selection.head = endpoint;
            cx.notify();
        }
        self.dragging_terminal = None;
    }

    fn copy_terminal_selection(&self, terminal_id: TerminalId, cx: &mut Context<Self>) -> bool {
        let Some(selection) = self.selection else {
            return false;
        };
        if selection.terminal_id != terminal_id {
            return false;
        }
        let Some(snapshot) = self.terminal_snapshot_for(terminal_id) else {
            return false;
        };
        if selection_bounds(snapshot, selection).is_none() {
            return false;
        }
        let text = selected_terminal_text(snapshot, selection);
        if text.is_empty() {
            return false;
        }
        cx.write_to_clipboard(gpui::ClipboardItem::new_string(text));
        true
    }

    fn terminal_resize_observer(
        &self,
        connection_id: ConnectionId,
        pane_id: PaneId,
        terminal_id: TerminalId,
        current_size: TerminalSize,
        metrics: TerminalMetrics,
        window_active: bool,
    ) -> AnyElement {
        // Capture the owning connection's transport at render time: the
        // window's active connection can change after this pane was drawn,
        // and a resize must always reach the server that owns the pane.
        let client = self
            .connection_by_id(connection_id)
            .map(|connection| connection.client.clone())
            .unwrap_or_else(|| self.client.clone());
        let resize_requests = self.resize_requests.clone();
        canvas(
            move |bounds, _, _| {
                // A terminal can be projected in several native windows. Only
                // the focused active window is allowed to negotiate its PTY
                // size, otherwise each window can fight over the shared size.
                if bounds.size.width <= px(0.) || bounds.size.height <= px(0.) {
                    return;
                }
                let size = TerminalSize::new(
                    terminal_columns_for_width(bounds, metrics),
                    terminal_lines_for_height(bounds, metrics),
                );
                let should_enqueue = {
                    let mut requests = resize_requests
                        .lock()
                        .expect("terminal resize requests poisoned");
                    terminal_resize_request_needed(
                        &mut requests,
                        connection_id,
                        pane_id,
                        terminal_id,
                        current_size,
                        size,
                        window_active,
                    )
                };
                if should_enqueue {
                    // The command must reach the server that owns this
                    // connection: dispatching through the window's
                    // active-connection client sent remote panes' resizes to
                    // the local model, which could not resolve the pane and
                    // dropped them.
                    if let Ok(operation_id) =
                        client.dispatch(AppCommand::Terminal(TerminalCommand::Resize {
                            terminal_id: None,
                            pane_id: Some(pane_id),
                            columns: size.columns,
                            lines: size.lines,
                        }))
                    {
                        let _ = client.wait_operation(operation_id);
                    }
                }
            },
            |_bounds, _, _, _| {},
        )
        .size_full()
        .absolute()
        .inset_0()
        .into_any_element()
    }

    fn handle_key_down(&mut self, event: &KeyDownEvent, cx: &mut Context<Self>) {
        if self.handle_dialog_key(event, cx) {
            cx.stop_propagation();
            return;
        }
        if self.context_menu.take().is_some() {
            cx.notify();
            cx.stop_propagation();
            return;
        }
        if self.handle_rename_key(event, cx) {
            cx.stop_propagation();
            return;
        }

        let keystroke = &event.keystroke;
        let shortcuts = &self.config.shortcuts;
        let focus_direction = [
            (&shortcuts.focus_left, "cmd-h", FocusDirection::Left),
            (&shortcuts.focus_right, "cmd-l", FocusDirection::Right),
            (&shortcuts.focus_up, "cmd-k", FocusDirection::Up),
            (&shortcuts.focus_down, "cmd-j", FocusDirection::Down),
        ]
        .into_iter()
        .find_map(|(shortcut, fallback, direction)| {
            shortcut_matches_or_default(shortcut, fallback, keystroke).then_some(direction)
        });
        if let Some(direction) = focus_direction {
            if self.focused_pane.is_some() {
                self.dispatch(
                    AppCommand::Pane(PaneCommand::Focus {
                        // Keep directional focus scoped to this window's
                        // selected workspace. With an explicit source pane,
                        // the dispatcher need not consult its global active
                        // workspace alias.
                        pane_id: self.focused_pane,
                        direction: Some(direction),
                    }),
                    cx,
                );
            }
            return;
        }
        if shortcut_matches_or_default(&shortcuts.close_pane, "cmd-shift-w", keystroke) {
            if let Some(pane_id) = self.focused_pane {
                self.dispatch(
                    AppCommand::Pane(PaneCommand::Close {
                        pane_id: Some(pane_id),
                    }),
                    cx,
                );
            }
            return;
        }
        if shortcut_matches_or_default(&shortcuts.paste, "cmd-v", keystroke) {
            if let Some(terminal_id) = self.active_terminal_id() {
                let modes = self
                    .active_terminal_snapshot()
                    .map(|snapshot| snapshot.modes)
                    .unwrap_or_default();
                self.focus_terminal_live_bottom(terminal_id, cx);
                self.paste_into_terminal(
                    terminal_id,
                    modes.bracketed_paste && self.config.features.bracketed_paste,
                    cx,
                );
            }
            return;
        }
        if shortcut_matches_or_default(&shortcuts.copy_or_interrupt, "cmd-c", keystroke) {
            if let Some(terminal_id) = self.active_terminal_id()
                && !self.copy_terminal_selection(terminal_id, cx)
            {
                self.focus_terminal_live_bottom(terminal_id, cx);
                self.enqueue_terminal_command(
                    terminal_id,
                    TerminalCommand::SendText {
                        terminal_id: Some(terminal_id),
                        pane_id: None,
                        text: "\u{3}".to_owned(),
                    },
                );
            }
            return;
        }
        if shortcut_matches_or_default(&shortcuts.eof, "cmd-d", keystroke) {
            if let Some(terminal_id) = self.active_terminal_id() {
                self.focus_terminal_live_bottom(terminal_id, cx);
                self.enqueue_terminal_command(
                    terminal_id,
                    TerminalCommand::SendText {
                        terminal_id: Some(terminal_id),
                        pane_id: None,
                        text: "\u{4}".to_owned(),
                    },
                );
            }
            return;
        }

        let Some(terminal_id) = self.active_terminal_id() else {
            return;
        };
        let modes = self
            .active_terminal_snapshot()
            .map(|snapshot| snapshot.modes)
            .unwrap_or_default();
        if shortcut_matches_or_default(&shortcuts.scroll_page_up, "shift-pageup", keystroke)
            || shortcut_matches_or_default(&shortcuts.scroll_page_down, "shift-pagedown", keystroke)
        {
            let lines = if shortcut_matches_or_default(
                &shortcuts.scroll_page_up,
                "shift-pageup",
                keystroke,
            ) {
                20
            } else {
                -20
            };
            self.scroll_accumulators.remove(&terminal_id);
            self.apply_local_viewport(terminal_id, None, Some(lines), cx);
            return;
        }
        let Some(text) = terminal_input_for_keystroke_with_modes(keystroke, modes) else {
            return;
        };
        if terminal_key_uses_text_input_handler(keystroke) && self.input_handler_terminal.is_some()
        {
            // GPUI dispatches the key event before forwarding printable text
            // to the focused InputHandler. Let the handler own it so a
            // printable character is never sent to the PTY twice.
            return;
        }
        let special_name =
            terminal_special_key_input_with_modes(&keystroke.key, keystroke.modifiers, modes)
                .is_some();
        // Like every other input path, a keystroke returns the viewport to
        // the live bottom (focus_terminal_live_bottom below). While Pi's
        // regular TUI is running, its output-driven redraws use the guarded
        // variant in apply_terminal_events, so the user's browsing position
        // is not moved by Pi's own output.
        self.focus_terminal_live_bottom(terminal_id, cx);
        self.clear_ime();
        self.enqueue_terminal_command(
            terminal_id,
            TerminalCommand::SendText {
                terminal_id: Some(terminal_id),
                pane_id: None,
                text,
            },
        );
        if special_name {
            // Synthesized keystrokes (water ctl / scenario control) carry the
            // key name itself in `key_char`. The mapping above already sent
            // the real byte sequence; without stopping propagation GPUI's
            // dispatch fallback would also insert the literal word.
            cx.stop_propagation();
        }
    }

    fn paste_into_terminal(
        &self,
        terminal_id: TerminalId,
        bracketed_paste: bool,
        cx: &mut Context<Self>,
    ) {
        let client = self
            .connection_by_id(self.active_connection)
            .map(|connection| connection.client.clone())
            .unwrap_or_else(|| self.client.clone());
        let clipboard = cx.read_from_clipboard_async();
        cx.spawn(async move |_entity, _cx| {
            let Ok(Some(item)) = clipboard.await else {
                return;
            };
            let Some(text) = clipboard_text(item) else {
                return;
            };
            let text = if bracketed_paste {
                format!("\u{1b}[200~{text}\u{1b}[201~")
            } else {
                text
            };
            let _ = client.enqueue(AppCommand::Terminal(TerminalCommand::SendText {
                terminal_id: Some(terminal_id),
                pane_id: None,
                text,
            }));
        })
        .detach();
    }

    pub(crate) fn install_snapshot(&mut self, snapshot: ModelSnapshot, cx: &mut Context<Self>) {
        if snapshot.state_revision <= self.snapshot.state_revision {
            return;
        }
        if let Some(connection) = self
            .connections
            .iter_mut()
            .find(|connection| connection.id == self.active_connection)
        {
            connection.snapshot = snapshot.clone();
        }
        if let Some(selection) = &self.selection
            && terminal_projection_in_snapshot(&snapshot, selection.terminal_id).is_none()
        {
            self.selection = None;
        }
        let previous_selection = self.selected_workspace;
        let previous_focused_pane = self.focused_pane;
        self.selected_workspace =
            workspace_selection_after_snapshot(self.selected_workspace, &snapshot);
        self.focused_pane =
            focused_pane_for_workspace(&snapshot, self.selected_workspace, self.focused_pane);
        let selection_moved = self.selected_workspace != previous_selection
            || self.focused_pane != previous_focused_pane;
        let active_connection = self.active_connection;
        self.collapsed_workspaces
            .retain(|(connection_id, workspace_id)| {
                *connection_id != active_connection
                    || workspace_exists_in_snapshot(&snapshot, *workspace_id)
            });
        self.snapshot = snapshot;
        // A snapshot that moved the visible tab away from a split under
        // drag (tab closed/activated elsewhere mid-drag) cancels the drag.
        if self.split_drag.as_ref().is_some_and(|drag| {
            self.selected_workspace_dump()
                .and_then(|workspace| workspace.active_tab)
                != Some(drag.tab_id)
        }) {
            self.split_drag = None;
        }
        let tab_strip_signature = (
            self.selected_workspace,
            self.selected_workspace_dump()
                .map_or(0, |workspace| workspace.tabs.len()),
        );
        if tab_strip_signature != self.last_tab_strip_signature || selection_moved {
            // The overflow state of the tab strip is only known after the
            // layout that follows this render; one follow-up render keeps
            // the edge indicators honest when tabs are added or removed.
            self.last_tab_strip_signature = tab_strip_signature;
            cx.notify();
        }
        // Reveal the selected tab whenever selection moves (new tab, tab
        // activation, Cmd+digit, workspace switch): gpui aligns the item
        // during the next prepaint using this frame's measured child
        // bounds and then clears the request, so manual scrolling is never
        // pinned afterwards.
        let active_tab = self
            .selected_workspace_dump()
            .and_then(|workspace| workspace.active_tab);
        if active_tab != self.last_revealed_tab {
            self.last_revealed_tab = active_tab;
            if let Some(index) = active_tab.and_then(|tab_id| {
                self.selected_workspace_dump().and_then(|workspace| {
                    workspace
                        .tabs
                        .iter()
                        .position(|tab| tab.id == tab_id)
                        // Reveal the trailing new-tab button as well when
                        // the last tab is selected, so "+" stays visible.
                        .map(|index| {
                            if index + 1 == workspace.tabs.len() {
                                index + 1
                            } else {
                                index
                            }
                        })
                })
            }) {
                self.tab_scroll.scroll_to_item(index);
                cx.notify();
            }
        }
        if let Some(pending) = self.pending_tab {
            let confirmed = self
                .selected_workspace_dump()
                .is_some_and(|workspace| workspace.active_tab == Some(pending));
            if confirmed || self.tab_by_id(pending).is_none() {
                self.pending_tab = None;
            }
        }
        let installed_snapshot = &self.snapshot;
        self.scroll_accumulators.retain(|terminal_id, state| {
            let Some(projection) =
                terminal_projection_in_snapshot(installed_snapshot, *terminal_id)
            else {
                return false;
            };
            if let Some(snapshot) = self.terminal_snapshots.get(terminal_id) {
                reconcile_visual_scroll(state, snapshot);
            }
            projection.summary.terminal_id == *terminal_id
        });
        self.mouse_scroll_animations
            .retain(|terminal_id, animation| {
                let Some(snapshot) = self.terminal_snapshots.get(terminal_id) else {
                    return false;
                };
                let base = snapshot.viewport_position as f32;
                !((snapshot.rows_before.is_empty() && animation.target_position > base)
                    || (snapshot.rows_after.is_empty() && animation.target_position < base))
            });
        self.pending_viewport_requests
            .retain(|terminal_id, (_, target)| {
                let Some(snapshot) = self.terminal_snapshots.get(terminal_id) else {
                    return false;
                };
                !((snapshot.rows_before.is_empty() && *target > snapshot.viewport_position)
                    || (snapshot.rows_after.is_empty() && *target < snapshot.viewport_position))
            });
        if self
            .context_menu
            .is_some_and(|menu| !self.context_menu_target_exists(menu.target))
        {
            self.context_menu = None;
        }
        if self
            .rename_target
            .is_some_and(|target| !self.rename_target_exists(target))
        {
            self.rename_target = None;
            self.rename_value.clear();
        }
        if self
            .dialog
            .is_some_and(|dialog| !self.dialog_target_exists(dialog))
        {
            self.dialog = None;
        }
        if self.ime_terminal.is_some() && self.ime_terminal != self.active_terminal_id() {
            self.clear_ime();
        }
        cx.notify();
    }

    fn dispatch_new_workspace(&mut self, cx: &mut Context<Self>) {
        let client = self.client.clone();
        cx.spawn(async move |entity, cx| {
            let result: Result<
                Option<(ModelSnapshot, WorkspaceId)>,
                crate::command::DispatchError,
            > = cx
                .background_executor()
                .spawn(async move {
                    let operation_id =
                        client.dispatch(AppCommand::Workspace(WorkspaceCommand::New))?;
                    let operation = client.wait_operation(operation_id)?;
                    let workspace_id = match operation.result {
                        Some(OperationResult::WorkspaceCreated { workspace_id }) => workspace_id,
                        _ => {
                            if operation.status.is_terminal() && operation.error.is_none() {
                                tracing::warn!(
                                    target: "water::ui",
                                    "new workspace completed without a workspace result"
                                );
                            } else {
                                tracing::warn!(
                                    target: "water::ui",
                                    status = ?operation.status,
                                    error = ?operation.error,
                                    "new workspace command failed"
                                );
                            }
                            return Ok(None);
                        }
                    };
                    if !operation.status.is_terminal() || operation.error.is_some() {
                        tracing::warn!(
                            target: "water::ui",
                            status = ?operation.status,
                            error = ?operation.error,
                            "new workspace command failed"
                        );
                        return Ok(None);
                    }
                    Ok(Some((client.state_dump()?, workspace_id)))
                })
                .await;
            if let Ok(Some((snapshot, workspace_id))) = result {
                let _ = entity.update(cx, |view, cx| {
                    view.install_snapshot(snapshot, cx);
                    view.select_workspace_locally(workspace_id, cx);
                });
            } else if let Err(error) = result {
                tracing::warn!(
                    target: "water::ui",
                    ?error,
                    "could not complete new workspace command"
                );
            }
        })
        .detach();
    }

    fn apply_operation_result(&mut self, result: OperationResult, cx: &mut Context<Self>) {
        match result {
            OperationResult::PaneCreated { pane_id } | OperationResult::PaneFocused { pane_id } => {
                if self
                    .selected_workspace_dump()
                    .is_some_and(|workspace| workspace_active_pane(workspace) == Some(pane_id))
                {
                    // Mouse-down begins the selection locally and then
                    // dispatches the focus command; when the operation
                    // result comes back, a selection that belongs to this
                    // pane's terminal is the in-progress drag, not stale
                    // state. Only drop selections from a different terminal
                    // (focus moved to another pane/terminal, or the pane
                    // has no terminal at all).
                    let pane_terminal = self.terminal_id_for_pane(pane_id);
                    let selection_belongs_to_pane = self
                        .selection
                        .as_ref()
                        .is_some_and(|selection| pane_terminal == Some(selection.terminal_id));
                    let was_focused = self.focused_pane == Some(pane_id);
                    self.focused_pane = Some(pane_id);
                    if !selection_belongs_to_pane {
                        self.selection = None;
                    }
                    if !was_focused || !selection_belongs_to_pane {
                        self.clear_ime();
                    }
                    cx.notify();
                }
            }
            OperationResult::PaneClosed { .. } | OperationResult::TabClosed { .. } => {
                self.focused_pane = focused_pane_for_workspace(
                    &self.snapshot,
                    self.selected_workspace,
                    self.focused_pane,
                );
                self.selection = None;
                self.clear_ime();
                cx.notify();
            }
            OperationResult::TabActivated { tab_id } | OperationResult::TabCreated { tab_id } => {
                if let Some(workspace) = self.selected_workspace_dump()
                    && workspace.active_tab == Some(tab_id)
                    && let Some(tab) = workspace.tabs.iter().find(|tab| tab.id == tab_id)
                {
                    self.focused_pane = Some(tab.active_pane);
                    self.selection = None;
                    self.clear_ime();
                    cx.notify();
                }
            }
            OperationResult::WorkspaceActivated { workspace_id } => {
                if self.apply_workspace_activated_locally(workspace_id) {
                    cx.notify();
                }
            }
            OperationResult::WorkspaceClosed { .. }
            | OperationResult::WorkspaceCreated { .. }
            | OperationResult::WorkspaceRenamed { .. }
            | OperationResult::PaneResized { .. }
            | OperationResult::SurfaceReplaced { .. }
            | OperationResult::TerminalSpawned { .. }
            | OperationResult::TerminalTextSent { .. }
            | OperationResult::TerminalBytesSent { .. }
            | OperationResult::TerminalResized { .. }
            | OperationResult::TerminalScrolled { .. }
            | OperationResult::TerminalViewportPositionSet { .. }
            | OperationResult::None
            | OperationResult::TabRenamed { .. } => {}
        }
    }

    fn dispatch(&mut self, command: AppCommand, cx: &mut Context<Self>) {
        self.dispatch_on(self.active_connection, command, cx);
    }

    /// Optimistically applies the `workspace.activate` effect locally.
    /// Activation is idempotent, so this is safe whether or not the
    /// dispatched command changes anything on the server; it also makes the
    /// switch instant on remote transports where the round trip takes longer
    /// than the user's next click. Only touches state this window owns.
    fn apply_workspace_activated_locally(&mut self, workspace_id: WorkspaceId) -> bool {
        if self
            .selected_workspace
            .is_some_and(|selected| selected == workspace_id)
        {
            return false;
        }
        let Some(workspace) = self.workspace_by_id(workspace_id) else {
            return false;
        };
        let focused_pane = workspace_active_pane(workspace);
        let changed = self.focused_pane != focused_pane;
        self.selected_workspace = Some(workspace_id);
        self.focused_pane = focused_pane;
        self.pending_tab = None;
        self.split_drag = None;
        self.selection = None;
        self.clear_ime();
        changed
    }

    /// Runs a command through the transport of `connection_id`. Used for
    /// sidebar actions that target a non-active connection (for example
    /// renaming a remote workspace); the resulting snapshot is installed on
    /// that connection's projection instead of the active one.
    fn dispatch_on(
        &mut self,
        connection_id: ConnectionId,
        command: AppCommand,
        cx: &mut Context<Self>,
    ) {
        let Some(client) = self
            .connection_by_id(connection_id)
            .map(|connection| connection.client.clone())
        else {
            cx.notify();
            return;
        };
        let command_name = command.type_name();
        cx.spawn(async move |entity, cx| {
            let result: Result<
                Option<(ModelSnapshot, OperationResult)>,
                crate::command::DispatchError,
            > = cx
                .background_executor()
                .spawn(async move {
                    let operation_id = client.dispatch(command)?;
                    let operation = client.wait_operation(operation_id)?;
                    if operation.status.is_terminal() && operation.error.is_none() {
                        Ok(Some((
                            client.state_dump()?,
                            operation.result.unwrap_or(OperationResult::None),
                        )))
                    } else {
                        tracing::warn!(
                            target: "water::ui",
                            command = command_name,
                            status = ?operation.status,
                            error = ?operation.error,
                            "UI command failed"
                        );
                        Ok(None)
                    }
                })
                .await;
            if let Ok(Some((snapshot, operation_result))) = result {
                let _ = entity.update(cx, |view, cx| {
                    if view.active_connection == connection_id {
                        view.install_snapshot(snapshot, cx);
                        view.apply_operation_result(operation_result, cx);
                    } else {
                        view.install_connection_snapshot(connection_id, snapshot, cx);
                    }
                });
            } else if let Err(error) = result {
                tracing::warn!(
                    target: "water::ui",
                    command = command_name,
                    ?error,
                    "could not complete UI command"
                );
            }
        })
        .detach();
    }

    fn tab_button(
        &self,
        tab_id: TabId,
        title: String,
        active: bool,
        active_pane: PaneId,
        theme: ThemeColors,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let background = if active {
            rgb(theme.tab_active_background)
        } else {
            rgb(theme.tab_inactive_background)
        };
        let title = if self.rename_target == Some(RenameTarget::Tab(tab_id)) {
            format!("{}▌", self.rename_value)
        } else {
            title
        };
        div()
            .id(format!("tab-{tab_id}"))
            .h(px(self.config.ui.tab_height))
            .px(px(10.))
            .items_center()
            .flex()
            .flex_none()
            .cursor_pointer()
            .hover(|style| style.bg(rgb(theme.tab_add_background)))
            .bg(background)
            .rounded(px(6.))
            .text_color(rgb(theme.terminal_foreground))
            .child(SharedString::from(title))
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(move |this, _event: &MouseDownEvent, window, cx| {
                    if this.has_transient_ui() {
                        cx.stop_propagation();
                        return;
                    }
                    this.context_menu = None;
                    this.focus_handle.focus(window, cx);
                    this.focused_pane = Some(active_pane);
                    this.split_drag = None;
                    this.selection = None;
                    this.clear_ime();
                    this.dispatch(
                        AppCommand::Tab(TabCommand::Activate {
                            tab_id: Some(tab_id),
                            index: None,
                        }),
                        cx,
                    );
                    cx.stop_propagation();
                }),
            )
            .on_mouse_down(
                MouseButton::Right,
                cx.listener(move |this, event: &MouseDownEvent, window, cx| {
                    this.context_menu = Some(ContextMenuState {
                        target: ContextMenuTarget::Tab(tab_id),
                        position: event.position,
                    });
                    this.focus_handle.focus(window, cx);
                    cx.stop_propagation();
                    cx.notify();
                }),
            )
            .into_any_element()
    }

    fn sidebar_resize_handle(&self, theme: ThemeColors, cx: &mut Context<Self>) -> AnyElement {
        div()
            .w(px(self.config.ui.sidebar_resize_handle_width))
            .h_full()
            .cursor(CursorStyle::ResizeLeftRight)
            .bg(rgb(theme.sidebar_background))
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(|this, _event: &MouseDownEvent, window, cx| {
                    this.dragging_sidebar = true;
                    this.focus_handle.focus(window, cx);
                    cx.stop_propagation();
                    cx.notify();
                }),
            )
            .into_any_element()
    }

    fn render_sidebar_workspace(
        &self,
        connection_id: ConnectionId,
        workspace: &WorkspaceDump,
        agents: &[&AgentDump],
        active_workspace: Option<WorkspaceId>,
        theme: ThemeColors,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let workspace_id = workspace.id;
        let active =
            self.active_connection == connection_id && active_workspace == Some(workspace_id);
        let collapsed = self
            .collapsed_workspaces
            .contains(&(connection_id, workspace_id));
        let running_agent_count = agents
            .iter()
            .filter(|agent| matches!(agent.status, crate::surface::TerminalStatus::Running))
            .count();
        let workspace_background = if active {
            rgb(theme.sidebar_workspace_active_background)
        } else {
            rgb(theme.sidebar_workspace_background)
        };
        let workspace_hover_background = if active {
            theme.sidebar_workspace_active_background
        } else {
            theme.tab_add_background
        };
        let workspace_active_pane = workspace
            .active_tab
            .and_then(|tab_id| workspace.tabs.iter().find(|tab| tab.id == tab_id))
            .map(|tab| tab.active_pane);
        let workspace_activate = cx.listener(move |this, event: &MouseDownEvent, window, cx| {
            this.context_menu = None;
            this.focus_handle.focus(window, cx);
            this.select_connection_locally(connection_id, cx);
            this.select_workspace_locally(workspace_id, cx);
            this.focused_pane = workspace_active_pane;
            this.begin_sidebar_drag(SidebarDragSource::Workspace(workspace_id), event.position);
            this.dispatch(
                AppCommand::Workspace(WorkspaceCommand::Activate {
                    workspace_id: Some(workspace_id),
                }),
                cx,
            );
            cx.stop_propagation();
        });
        let disclosure = div()
            .id(format!(
                "workspace-disclosure-{connection_id}-{workspace_id}"
            ))
            .w(px(SIDEBAR_DISCLOSURE_WIDTH))
            .flex_shrink_0()
            .items_center()
            .justify_center()
            .flex()
            .cursor_pointer()
            .text_color(rgb(theme.ui_foreground))
            .child(SharedString::from(if collapsed { "▸" } else { "▾" }))
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(move |this, _event: &MouseDownEvent, _window, cx| {
                    this.toggle_workspace_collapsed(connection_id, workspace_id, cx);
                    cx.stop_propagation();
                }),
            );
        let mut workspace_row = div()
            .id(format!("workspace-{connection_id}-{workspace_id}"))
            .h(px(self.config.ui.sidebar_header_height))
            .w_full()
            .px(px(10.))
            .items_center()
            .flex()
            .cursor_pointer()
            .hover(|style| style.bg(rgb(workspace_hover_background)))
            .bg(workspace_background)
            .text_color(rgb(theme.ui_foreground))
            .on_mouse_down(MouseButton::Left, workspace_activate)
            .on_mouse_down(
                MouseButton::Right,
                cx.listener(move |this, event: &MouseDownEvent, window, cx| {
                    this.select_connection_locally(connection_id, cx);
                    this.context_menu = Some(ContextMenuState {
                        target: ContextMenuTarget::Workspace {
                            connection_id,
                            workspace_id,
                        },
                        position: event.position,
                    });
                    this.focus_handle.focus(window, cx);
                    cx.stop_propagation();
                    cx.notify();
                }),
            )
            .child(disclosure)
            .child(
                div()
                    .flex_1()
                    .min_w(px(0.))
                    .truncate()
                    .child(SharedString::from(workspace.title.clone())),
            );
        if self.config.ui.sidebar_show_agent_count && running_agent_count > 0 {
            workspace_row = workspace_row.child(
                div()
                    .flex_shrink_0()
                    .text_color(rgb(theme.inactive_pane_border))
                    .child(SharedString::from(running_agent_count.to_string())),
            );
        }

        let mut group = div()
            .id(format!("workspace-group-{connection_id}-{workspace_id}"))
            .w_full()
            .flex()
            .flex_col()
            .child(workspace_row);
        if !collapsed {
            for agent in agents {
                group = group.child(self.render_sidebar_agent(connection_id, agent, theme, cx));
            }
        }
        group.into_any_element()
    }

    fn render_sidebar_connection(
        &self,
        connection: &WorkspaceConnection,
        theme: ThemeColors,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let connection_id = connection.id;
        let collapsed = self.collapsed_connections.contains(&connection_id);
        let selected = self.active_connection == connection_id;
        let title = connection.title.clone();
        let kind_label = match connection.kind {
            WorkspaceConnectionKind::Local => "LOCAL",
            WorkspaceConnectionKind::Remote => "SSH",
        };
        let mut header = div()
            .id(format!("connection-{connection_id}"))
            .h(px(34.))
            .w_full()
            .px(px(10.))
            .items_center()
            .flex()
            .flex_none()
            .cursor_pointer()
            .bg(rgb(if selected {
                theme.sidebar_connection_active_background
            } else {
                theme.sidebar_connection_background
            }))
            .hover(|style| style.bg(rgb(theme.tab_add_background)))
            .child(
                div()
                    .flex_1()
                    .min_w(px(0.))
                    .truncate()
                    .child(SharedString::from(title)),
            )
            .child(
                div()
                    .flex_shrink_0()
                    .text_size(px(9.))
                    .text_color(rgb(theme.inactive_pane_border))
                    .child(kind_label),
            )
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(move |this, _event: &MouseDownEvent, window, cx| {
                    this.context_menu = None;
                    this.focus_handle.focus(window, cx);
                    this.select_connection_locally(connection_id, cx);
                    this.toggle_connection_collapsed(connection_id, cx);
                    cx.stop_propagation();
                }),
            );
        if connection.kind == WorkspaceConnectionKind::Remote {
            header = header.on_mouse_down(
                MouseButton::Right,
                cx.listener(move |this, event: &MouseDownEvent, window, cx| {
                    this.context_menu = Some(ContextMenuState {
                        target: ContextMenuTarget::Connection(connection_id),
                        position: event.position,
                    });
                    this.focus_handle.focus(window, cx);
                    cx.stop_propagation();
                    cx.notify();
                }),
            );
        }

        let mut group = div()
            .id(format!("connection-group-{connection_id}"))
            .w_full()
            .flex()
            .flex_col()
            .child(header);
        if collapsed {
            return group.into_any_element();
        }

        let active_workspace = if selected {
            self.active_workspace_id()
        } else {
            None
        };
        let workspaces = if connection.snapshot.workspaces.is_empty() {
            connection.snapshot.workspace.iter().collect::<Vec<_>>()
        } else {
            connection.snapshot.workspaces.iter().collect::<Vec<_>>()
        };
        if workspaces.is_empty() {
            group = group.child(
                div()
                    .h(px(24.))
                    .w_full()
                    .pl(px(34.))
                    .items_center()
                    .flex()
                    .text_color(rgb(theme.inactive_pane_border))
                    .child("No workspaces"),
            );
        }
        for workspace in workspaces {
            let agents = connection
                .snapshot
                .agents
                .iter()
                .filter(|agent| agent.workspace_id == workspace.id)
                .collect::<Vec<_>>();
            group = group.child(self.render_sidebar_workspace(
                connection_id,
                workspace,
                &agents,
                active_workspace,
                theme,
                cx,
            ));
        }
        group.into_any_element()
    }

    fn render_sidebar(&self, theme: ThemeColors, cx: &mut Context<Self>) -> AnyElement {
        let mut list = div()
            .id("workspace-list")
            .flex_1()
            .min_h(px(0.))
            .overflow_y_scroll()
            .track_scroll(&self.sidebar_scroll)
            .flex_col();
        for connection in &self.connections {
            list = list.child(self.render_sidebar_connection(connection, theme, cx));
        }
        let list_bounds = self.sidebar_scroll.bounds();
        let indicator = self.sidebar_drop_preview.and_then(|preview| {
            let y = match preview {
                SidebarDropPreview::Workspace { y, .. } | SidebarDropPreview::Agent { y, .. } => y,
            };
            if list_bounds.size.height <= px(0.) {
                return None;
            }
            let top = (y - f32::from(list_bounds.origin.y))
                .clamp(0.0, (f32::from(list_bounds.size.height) - 2.0).max(0.0));
            Some(
                canvas(
                    |_bounds, _, _| (),
                    move |bounds, _, window, _| {
                        window.paint_quad(fill(bounds, rgb(theme.sidebar_drag_indicator)));
                    },
                )
                .absolute()
                .left_0()
                .right_0()
                .top(px(top))
                .h(px(2.0)),
            )
        });
        let connect_remote = div()
            .id("connect-remote")
            .h(px(32.))
            .w_full()
            .px(px(10.))
            .items_center()
            .gap(px(6.))
            .flex()
            .flex_none()
            .cursor_pointer()
            .border_t_1()
            .border_color(rgb(theme.inactive_pane_border))
            .hover(|style| style.bg(rgb(theme.tab_add_background)))
            .child("＋")
            .child("Connect Remote…")
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(|this, _event: &MouseDownEvent, window, cx| {
                    this.begin_connect_remote(window, cx);
                    cx.stop_propagation();
                }),
            );
        let mut sidebar_content = div()
            .flex_1()
            .min_w(px(0.))
            .h_full()
            .flex()
            .flex_col()
            .relative()
            .child(list)
            .child(connect_remote);
        if let Some(indicator) = indicator {
            sidebar_content = sidebar_content.child(indicator);
        }
        div()
            .w(px(self.sidebar_width))
            .h_full()
            .flex()
            .flex_row()
            .bg(rgb(theme.sidebar_background))
            .text_color(rgb(theme.ui_foreground))
            .child(sidebar_content)
            .child(self.sidebar_resize_handle(theme, cx))
            .into_any_element()
    }

    /// One row in the sidebar. Clicking activates the pane's full location
    /// through the regular command path (`pane.focus` re-activates the owning
    /// workspace and tab in the model).
    fn render_sidebar_agent(
        &self,
        connection_id: ConnectionId,
        agent: &AgentDump,
        theme: ThemeColors,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let workspace_id = agent.workspace_id;
        let pane_id = agent.pane_id;
        let focused_here = self.active_connection == connection_id
            && self.selected_workspace == Some(workspace_id)
            && self.focused_pane == Some(pane_id);
        let row_background = if focused_here {
            rgb(theme.sidebar_agent_active_background)
        } else {
            rgb(theme.sidebar_agent_background)
        };
        let hover_background = if focused_here {
            theme.sidebar_agent_active_background
        } else {
            theme.tab_add_background
        };
        // The configured per-kind color is the row's identity: full strength
        // while the agent is producing output, dimmed toward the row
        // background when idle, and neutral once the process exited. A
        // focused row paints on its configured active background, so a kind
        // color too close to that background is blended toward the terminal
        // background to stay readable.
        let kind_color = theme.agent_color(agent.kind);
        let running = matches!(agent.status, crate::surface::TerminalStatus::Running);
        let mut dot_color = if !running {
            theme.inactive_pane_border
        } else if agent.active {
            kind_color
        } else {
            let background = if focused_here {
                theme.sidebar_agent_active_background
            } else {
                theme.sidebar_agent_background
            };
            mix_rgb(kind_color, background, 0.45)
        };
        if running
            && focused_here
            && channel_distance(dot_color, theme.sidebar_agent_active_background) < 0x30
        {
            dot_color = mix_rgb(dot_color, theme.terminal_background, 0.5);
        }
        let project = agent
            .cwd
            .rsplit('/')
            .find(|segment| !segment.is_empty())
            .unwrap_or("")
            .to_owned();
        let agent_label = if self.rename_target
            == Some(RenameTarget::Agent {
                connection_id,
                pane_id,
            }) {
            format!("{}▌", self.rename_value)
        } else {
            agent.display_label().to_owned()
        };
        let agent_activate = cx.listener(move |this, event: &MouseDownEvent, window, cx| {
            this.context_menu = None;
            this.focus_handle.focus(window, cx);
            this.select_connection_locally(connection_id, cx);
            this.select_workspace_locally(workspace_id, cx);
            this.focused_pane = Some(pane_id);
            this.begin_sidebar_drag(
                SidebarDragSource::Agent {
                    pane_id,
                    workspace_id,
                },
                event.position,
            );
            this.dispatch(
                AppCommand::Pane(PaneCommand::Focus {
                    pane_id: Some(pane_id),
                    direction: None,
                }),
                cx,
            );
            cx.stop_propagation();
        });
        let mut row = div()
            .id(format!("agent-pane-{connection_id}-{pane_id}"))
            .h(px(28.))
            .w_full()
            .px(px(10.))
            .items_center()
            .gap(px(6.))
            .flex()
            .cursor_pointer()
            .hover(move |style| style.bg(rgb(hover_background)))
            .bg(row_background)
            .text_color(rgb(theme.ui_foreground))
            .on_mouse_down(MouseButton::Left, agent_activate)
            .child(div().w(px(SIDEBAR_DISCLOSURE_WIDTH)).flex_shrink_0())
            .child(
                div()
                    .flex_shrink_0()
                    .text_color(rgb(dot_color))
                    .child(SharedString::from("●")),
            )
            .child(SharedString::from(agent_label))
            .child(
                div()
                    .flex_1()
                    .min_w(px(0.))
                    .text_right()
                    .truncate()
                    .text_color(rgb(theme.inactive_pane_border))
                    .child(SharedString::from(project)),
            );
        if matches!(agent.status, crate::surface::TerminalStatus::Running) {
            row = row.on_mouse_down(
                MouseButton::Right,
                cx.listener(move |this, event: &MouseDownEvent, window, cx| {
                    this.select_connection_locally(connection_id, cx);
                    this.context_menu = Some(ContextMenuState {
                        target: ContextMenuTarget::Agent {
                            connection_id,
                            pane_id,
                        },
                        position: event.position,
                    });
                    this.focus_handle.focus(window, cx);
                    cx.stop_propagation();
                    cx.notify();
                }),
            );
        }
        row.into_any_element()
    }

    fn render_context_menu_item(
        &self,
        label: &'static str,
        theme: ThemeColors,
        listener: impl Fn(&mut Self, &MouseDownEvent, &mut Window, &mut Context<Self>) + 'static,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        div()
            .h(px(30.))
            .w_full()
            .px(px(10.))
            .items_center()
            .flex()
            .cursor_pointer()
            .hover(|style| style.bg(rgb(theme.tab_inactive_background)))
            .text_color(rgb(theme.ui_foreground))
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(move |this, event, window, cx| {
                    listener(this, event, window, cx);
                    cx.stop_propagation();
                }),
            )
            .child(label)
            .into_any_element()
    }

    fn render_context_menu(
        &self,
        theme: ThemeColors,
        cx: &mut Context<Self>,
    ) -> Option<AnyElement> {
        let context_menu = self.context_menu?;
        let target = context_menu.target;
        let rename = match target {
            ContextMenuTarget::Connection(_) => None,
            ContextMenuTarget::Workspace {
                connection_id,
                workspace_id,
            } => Some(self.render_context_menu_item(
                "Rename workspace",
                theme,
                move |this, _event, window, cx| {
                    this.context_menu = None;
                    this.begin_rename_workspace(connection_id, workspace_id, window, cx);
                },
                cx,
            )),
            ContextMenuTarget::Tab(tab_id) => Some(self.render_context_menu_item(
                "Rename tab",
                theme,
                move |this, _event, window, cx| {
                    this.context_menu = None;
                    this.begin_rename_tab(tab_id, window, cx);
                },
                cx,
            )),
            ContextMenuTarget::Agent {
                connection_id,
                pane_id,
            } => Some(self.render_context_menu_item(
                "Rename agent",
                theme,
                move |this, _event, window, cx| {
                    this.context_menu = None;
                    this.begin_rename_agent(connection_id, pane_id, window, cx);
                },
                cx,
            )),
        };
        let close = match target {
            ContextMenuTarget::Connection(_) => None,
            ContextMenuTarget::Workspace {
                connection_id: _,
                workspace_id,
            } => Some(self.render_context_menu_item(
                "Close workspace",
                theme,
                move |this, _event, window, cx| {
                    this.request_close_workspace(workspace_id, window, cx);
                },
                cx,
            )),
            ContextMenuTarget::Tab(tab_id) => Some(self.render_context_menu_item(
                "Close tab",
                theme,
                move |this, _event, _window, cx| {
                    this.context_menu = None;
                    this.dispatch(
                        AppCommand::Tab(TabCommand::Close {
                            tab_id: Some(tab_id),
                        }),
                        cx,
                    );
                    cx.notify();
                },
                cx,
            )),
            ContextMenuTarget::Agent { .. } => None,
        };
        let mut menu = div()
            .id("workspace-context-menu")
            .w(px(190.))
            .p(px(4.))
            .flex()
            .flex_col()
            .gap(px(2.))
            .bg(rgb(theme.chrome_background))
            .border_1()
            .border_color(rgb(theme.inactive_pane_border))
            .rounded(px(12.));
        if let Some(rename) = rename {
            menu = menu.child(rename);
        }
        if let Some(close) = close {
            menu = menu.child(close);
        }
        if let ContextMenuTarget::Connection(connection_id) = target {
            menu = menu
                .child(self.render_context_menu_item(
                    "Disconnect",
                    theme,
                    move |this, _event, _window, cx| {
                        this.context_menu = None;
                        if let Some(application) = this.application.clone() {
                            application.disconnect_connection(connection_id, cx);
                        }
                    },
                    cx,
                ))
                .child(self.render_context_menu_item(
                    "Kill Server",
                    theme,
                    move |this, _event, _window, cx| {
                        this.context_menu = None;
                        if let Some(application) = this.application.clone() {
                            application.kill_connection(connection_id, cx);
                        }
                    },
                    cx,
                ));
        }
        let position = context_menu.position;
        Some(
            deferred(
                div()
                    .size_full()
                    .absolute()
                    .inset_0()
                    .on_mouse_down(
                        MouseButton::Left,
                        cx.listener(|this, _event: &MouseDownEvent, _window, cx| {
                            this.context_menu = None;
                            cx.stop_propagation();
                            cx.notify();
                        }),
                    )
                    .on_mouse_down(
                        MouseButton::Right,
                        cx.listener(|this, _event: &MouseDownEvent, _window, cx| {
                            this.context_menu = None;
                            cx.stop_propagation();
                            cx.notify();
                        }),
                    )
                    .child(
                        anchored()
                            .position(position)
                            .snap_to_window_with_margin(px(8.))
                            .child(menu),
                    ),
            )
            .with_priority(10)
            .into_any_element(),
        )
    }

    fn render_dialog(&self, theme: ThemeColors, cx: &mut Context<Self>) -> Option<AnyElement> {
        let dialog = self.dialog?;
        if dialog == DialogState::ConnectRemote {
            return Some(self.render_connect_remote_dialog(theme, cx));
        }
        if matches!(dialog, DialogState::RenameWorkspace { .. }) {
            return Some(self.render_rename_workspace_dialog(theme, cx));
        }
        let DialogState::ConfirmCloseWorkspace { workspace_id } = dialog else {
            unreachable!("text-input dialogs returned above")
        };
        let workspace = self.workspace_by_id(workspace_id)?;
        let tab_count = workspace.tabs.len();
        let tab_label = if tab_count == 1 { "tab" } else { "tabs" };
        let message = format!(
            "Close “{}” and its {tab_count} {tab_label}?",
            workspace.title
        );
        let cancel = div()
            .h(px(30.))
            .px(px(12.))
            .items_center()
            .justify_center()
            .flex()
            .text_color(rgb(theme.ui_foreground))
            .border_1()
            .border_color(rgb(theme.inactive_pane_border))
            .rounded(px(6.))
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(|this, _event: &MouseDownEvent, _window, cx| {
                    this.cancel_dialog(cx);
                    cx.stop_propagation();
                }),
            )
            .child("Cancel");
        let confirm = div()
            .h(px(30.))
            .px(px(12.))
            .items_center()
            .justify_center()
            .flex()
            .bg(rgb(theme.tab_add_background))
            .rounded(px(6.))
            .text_color(rgb(theme.ui_foreground))
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(|this, _event: &MouseDownEvent, _window, cx| {
                    this.confirm_dialog(cx);
                    cx.stop_propagation();
                }),
            )
            .child("Close workspace");
        let dialog = div()
            .id("close-workspace-dialog")
            .w(px(380.))
            .p(px(20.))
            .gap(px(12.))
            .flex()
            .flex_col()
            .bg(rgb(theme.chrome_background))
            .border_1()
            .border_color(rgb(theme.active_pane_border))
            .rounded(px(12.))
            .text_color(rgb(theme.ui_foreground))
            .child(
                div()
                    .text_size(px(self.config.ui.font_size * 1.125))
                    .child("Close workspace?"),
            )
            .child(SharedString::from(message))
            .child(
                div()
                    .w_full()
                    .gap(px(8.))
                    .justify_end()
                    .items_center()
                    .flex()
                    .child(cancel)
                    .child(confirm),
            );
        Some(
            deferred(
                div()
                    .debug_selector(|| "dialog-scrim".into())
                    .size_full()
                    .absolute()
                    .inset_0()
                    .flex()
                    .items_center()
                    .justify_center()
                    .bg(rgba(DIALOG_SCRIM))
                    .on_mouse_down(MouseButton::Left, |_event: &MouseDownEvent, _window, cx| {
                        cx.stop_propagation();
                    })
                    .on_mouse_down(
                        MouseButton::Right,
                        |_event: &MouseDownEvent, _window, cx| {
                            cx.stop_propagation();
                        },
                    )
                    .child(dialog),
            )
            .with_priority(20)
            .into_any_element(),
        )
    }

    fn render_connect_remote_dialog(
        &self,
        theme: ThemeColors,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let cancel = div()
            .h(px(30.))
            .px(px(12.))
            .items_center()
            .justify_center()
            .flex()
            .cursor_pointer()
            .text_color(rgb(theme.ui_foreground))
            .border_1()
            .border_color(rgb(theme.inactive_pane_border))
            .rounded(px(6.))
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(|this, _event: &MouseDownEvent, _window, cx| {
                    this.cancel_dialog(cx);
                    cx.stop_propagation();
                }),
            )
            .child("Cancel");
        let connect_label = if self.remote_connection_pending {
            "Connecting…"
        } else {
            "Connect"
        };
        let hint_text = if self.dialog_input.trim().is_empty() && !self.remote_connection_pending {
            "Enter an SSH host or config alias to continue"
        } else {
            ""
        };
        let mut dialog = div()
            .id("connect-remote-dialog")
            .w(px(420.))
            .p(px(20.))
            .gap(px(12.))
            .flex()
            .flex_col()
            .bg(rgb(theme.chrome_background))
            .border_1()
            .border_color(rgb(theme.active_pane_border))
            .rounded(px(12.))
            .text_color(rgb(theme.ui_foreground))
            .text_size(px(self.config.ui.font_size));
        if !self.config.ui.font_family.is_empty() {
            dialog = dialog.font(font(self.config.ui.font_family.clone()));
        }
        dialog = dialog
            .child(
                div()
                    .text_size(px(self.config.ui.font_size * 1.125))
                    .child("Connect to remote Water"),
            )
            .child("Uses your OpenSSH config and agent. The authenticated connection is reused.")
            .child(self.render_text_input_row(theme))
            .child(
                div()
                    .w_full()
                    .text_size(px(self.config.ui.font_size * 0.875))
                    .text_color(rgb(theme.inactive_pane_border))
                    .child(SharedString::from(hint_text.to_owned())),
            );
        if let Some(error) = &self.remote_connection_error {
            dialog = dialog.child(
                div()
                    .text_color(rgb(theme.terminal_foreground))
                    .child(SharedString::from(error.clone())),
            );
        }
        let button_row = div()
            .w_full()
            .gap(px(8.))
            .items_center()
            .justify_end()
            .flex()
            .child(div().flex_none().child(cancel))
            .child(
                div()
                    .flex_none()
                    .child(self.render_dialog_confirm_button(connect_label, theme, cx)),
            );
        dialog = dialog.child(button_row);
        deferred(
            div()
                .debug_selector(|| "dialog-scrim".into())
                .size_full()
                .absolute()
                .inset_0()
                .flex()
                .items_center()
                .justify_center()
                .bg(rgba(DIALOG_SCRIM))
                .on_mouse_down(MouseButton::Left, |_event: &MouseDownEvent, _window, cx| {
                    cx.stop_propagation();
                })
                .on_mouse_down(
                    MouseButton::Right,
                    |_event: &MouseDownEvent, _window, cx| {
                        cx.stop_propagation();
                    },
                )
                .child(dialog),
        )
        .with_priority(20)
        .into_any_element()
    }

    /// The bordered, editable row shared by the text-input dialogs. The
    /// caret is rendered as a separate opacity-toggled element so that
    /// blinking does not shift the text layout.
    fn render_text_input_row(&self, theme: ThemeColors) -> AnyElement {
        let value = &self.dialog_input;
        let caret = self.dialog_caret.clamp(0, value.len());
        let before = SharedString::from(value[..caret].to_owned());
        let after = SharedString::from(value[caret..].to_owned());
        let caret_opacity = if self.dialog_caret_visible { 1.0 } else { 0.0 };
        let mut row = div()
            .debug_selector(|| "dialog-text-input".into())
            .id("dialog-text-input")
            .h(px(34.))
            .w_full()
            .px(px(10.))
            .items_center()
            .flex()
            .border_1()
            .border_color(rgb(theme.inactive_pane_border))
            .rounded(px(6.))
            .text_size(px(self.config.ui.font_size))
            .text_color(rgb(theme.ui_foreground));
        if !self.config.ui.font_family.is_empty() {
            row = row.font(font(self.config.ui.font_family.clone()));
        }
        row.child(
            div()
                .w_full()
                .overflow_hidden()
                .flex()
                .child(before)
                .child(div().opacity(caret_opacity).child("▌"))
                .child(after),
        )
        .into_any_element()
    }

    /// Confirm button of a text-input dialog: highlighted while the input is
    /// valid, dimmed and inert while it is empty.
    fn render_dialog_confirm_button(
        &self,
        label: &str,
        theme: ThemeColors,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let enabled = self.dialog_input_is_valid();
        let mut button = div()
            .h(px(30.))
            .px(px(12.))
            .items_center()
            .justify_center()
            .flex()
            .rounded(px(6.))
            .bg(rgb(if enabled {
                theme.tab_add_background
            } else {
                theme.chrome_background
            }))
            .text_color(rgb(if enabled {
                theme.ui_foreground
            } else {
                theme.inactive_pane_border
            }));
        if enabled {
            button = button.cursor_pointer().on_mouse_down(
                MouseButton::Left,
                cx.listener(|this, _event: &MouseDownEvent, _window, cx| {
                    this.confirm_dialog(cx);
                    cx.stop_propagation();
                }),
            );
        }
        button
            .child(SharedString::from(label.to_owned()))
            .into_any_element()
    }

    fn render_rename_workspace_dialog(
        &self,
        theme: ThemeColors,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let Some(DialogState::RenameWorkspace {
            connection_id,
            workspace_id,
        }) = self.dialog
        else {
            return div().into_any_element();
        };
        let title = self
            .workspace_by_id_in(connection_id, workspace_id)
            .map(|workspace| workspace.title.clone())
            .unwrap_or_default();
        let cancel = div()
            .h(px(30.))
            .px(px(12.))
            .items_center()
            .justify_center()
            .flex()
            .cursor_pointer()
            .text_color(rgb(theme.ui_foreground))
            .border_1()
            .border_color(rgb(theme.inactive_pane_border))
            .rounded(px(6.))
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(|this, _event: &MouseDownEvent, _window, cx| {
                    this.cancel_dialog(cx);
                    cx.stop_propagation();
                }),
            )
            .child("Cancel");
        let hint_text = if self.dialog_input.trim().is_empty() {
            "Enter a name to rename this workspace"
        } else {
            ""
        };
        let mut dialog = div()
            .id("rename-workspace-dialog")
            .w(px(420.))
            .p(px(20.))
            .gap(px(12.))
            .flex()
            .flex_col()
            .bg(rgb(theme.chrome_background))
            .border_1()
            .border_color(rgb(theme.active_pane_border))
            .rounded(px(12.))
            .text_color(rgb(theme.ui_foreground))
            .text_size(px(self.config.ui.font_size));
        if !self.config.ui.font_family.is_empty() {
            dialog = dialog.font(font(self.config.ui.font_family.clone()));
        }
        dialog = dialog
            .child(
                div()
                    .text_size(px(self.config.ui.font_size * 1.125))
                    .child("Rename workspace"),
            )
            .child(
                div()
                    .text_size(px(self.config.ui.font_size * 0.875))
                    .text_color(rgb(theme.inactive_pane_border))
                    .child(SharedString::from(format!("Current name: {title}"))),
            )
            .child(self.render_text_input_row(theme))
            .child(
                div()
                    .w_full()
                    .text_size(px(self.config.ui.font_size * 0.875))
                    .text_color(rgb(theme.inactive_pane_border))
                    .child(SharedString::from(hint_text.to_owned())),
            );
        let button_row = div()
            .w_full()
            .gap(px(8.))
            .items_center()
            .justify_end()
            .flex()
            .child(div().flex_none().child(cancel))
            .child(
                div()
                    .flex_none()
                    .child(self.render_dialog_confirm_button("Rename", theme, cx)),
            );
        dialog = dialog.child(button_row);
        deferred(
            div()
                .debug_selector(|| "dialog-scrim".into())
                .size_full()
                .absolute()
                .inset_0()
                .flex()
                .items_center()
                .justify_center()
                .bg(rgba(DIALOG_SCRIM))
                .on_mouse_down(MouseButton::Left, |_event: &MouseDownEvent, _window, cx| {
                    cx.stop_propagation();
                })
                .on_mouse_down(
                    MouseButton::Right,
                    |_event: &MouseDownEvent, _window, cx| {
                        cx.stop_propagation();
                    },
                )
                .child(dialog),
        )
        .with_priority(20)
        .into_any_element()
    }

    fn render_titlebar_control(
        &self,
        color: u32,
        area: WindowControlArea,
        listener: impl Fn(&mut Self, &MouseDownEvent, &mut Window, &mut Context<Self>) + 'static,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        div()
            .size(px(12.))
            .rounded(px(6.))
            .bg(rgb(color))
            .cursor_pointer()
            .occlude()
            .window_control_area(area)
            .hover(|style| style.opacity(0.75))
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(move |this, event, window, cx| {
                    cx.stop_propagation();
                    listener(this, event, window, cx);
                }),
            )
            .into_any_element()
    }

    fn render_tab_bar(&self, theme: ThemeColors, cx: &mut Context<Self>) -> AnyElement {
        let tab_data: Vec<(TabId, String, bool, PaneId)> = self
            .selected_workspace_dump()
            .map(|workspace| {
                workspace
                    .tabs
                    .iter()
                    .map(|tab| {
                        (
                            tab.id,
                            tab.title.clone(),
                            workspace.active_tab == Some(tab.id),
                            tab.active_pane,
                        )
                    })
                    .collect()
            })
            .unwrap_or_default();
        let mut tab_strip = div()
            .id("tab-bar-scroll")
            .h_full()
            .w_full()
            .gap(px(2.))
            .items_center()
            .flex()
            .overflow_x_scroll()
            .restrict_scroll_to_axis()
            .track_scroll(&self.tab_scroll)
            // Tabs are DIRECT children of the tracked element so gpui's
            // `scroll_to_item` indices line up with tab indices. The
            // container's built-in handler scrolls on horizontal wheel
            // deltas; `restrict_scroll_to_axis` disables its vertical-to-
            // horizontal mapping so the vertical gesture stays entirely
            // with the opt-in gate below (no double application).
            .on_scroll_wheel(cx.listener(|this, event, _window, cx| {
                this.scroll_tab_bar(event, cx);
            }));
        for (tab_id, title, active, active_pane) in tab_data {
            tab_strip =
                tab_strip.child(self.tab_button(tab_id, title, active, active_pane, theme, cx));
        }
        // The new-tab button rides at the end of the scrollable strip so it
        // always sits next to the last tab; the edge indicators below signal
        // when the strip overflows.
        let new_tab = div()
            .id("new-tab")
            .h(px(self.config.ui.tab_height))
            .w(px(28.))
            .items_center()
            .justify_center()
            .flex()
            .flex_none()
            .cursor_pointer()
            .hover(|style| style.bg(rgb(theme.tab_add_background)))
            .bg(rgb(theme.tab_inactive_background))
            .rounded(px(6.))
            .text_color(rgb(theme.ui_foreground))
            .child("+")
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(|this, _event: &MouseDownEvent, window, cx| {
                    if this.has_transient_ui() {
                        cx.stop_propagation();
                        return;
                    }
                    this.focus_handle.focus(window, cx);
                    this.new_terminal_tab(cx);
                    cx.stop_propagation();
                }),
            );
        let tab_strip = tab_strip.child(new_tab);
        let tab_viewport = tab_strip;
        // The indicator state reflects the previous layout pass; scroll and
        // snapshot updates re-render and keep it current.
        let (show_left, show_right) = tab_bar_edge_indicators(
            f32::from(self.tab_scroll.max_offset().x),
            f32::from(self.tab_scroll.offset().x),
        );
        let indicator = |side: isize| {
            let glyph = if side < 0 { "\u{25c2}" } else { "\u{25b8}" };
            let step = side as f32 * TAB_SCROLL_NUDGE_PX;
            div()
                .id(if side < 0 {
                    "tab-scroll-left"
                } else {
                    "tab-scroll-right"
                })
                .absolute()
                .top_0()
                .h_full()
                .w(px(20.))
                .items_center()
                .justify_center()
                .flex()
                .cursor_pointer()
                .bg(rgb(theme.chrome_background))
                .text_color(rgb(theme.ui_foreground))
                .hover(|style| style.bg(rgb(theme.tab_add_background)))
                .child(glyph)
                .on_mouse_down(
                    MouseButton::Left,
                    cx.listener(move |this, _event: &MouseDownEvent, _window, cx| {
                        this.nudge_tab_bar_scroll(step, cx);
                        cx.stop_propagation();
                    }),
                )
                .on_scroll_wheel(cx.listener(move |this, event, _window, cx| {
                    this.scroll_tab_bar(event, cx);
                }))
        };
        let left_indicator = if show_left {
            Some(indicator(-1).left_0())
        } else {
            None
        };
        let right_indicator = if show_right {
            Some(indicator(1).right_0())
        } else {
            None
        };
        let mut root = div()
            .id("tab-bar")
            .h_full()
            .flex_1()
            .min_w(px(0.))
            .relative()
            .flex()
            .child(tab_viewport);
        if let Some(left_indicator) = left_indicator {
            root = root.child(left_indicator);
        }
        if let Some(right_indicator) = right_indicator {
            root = root.child(right_indicator);
        }
        root.into_any_element()
    }

    fn render_titlebar(&self, theme: ThemeColors, cx: &mut Context<Self>) -> AnyElement {
        let close = self.render_titlebar_control(
            0xff5f57,
            WindowControlArea::Close,
            |_, _event, window, _cx| window.remove_window(),
            cx,
        );
        let minimize = self.render_titlebar_control(
            0xfebc2e,
            WindowControlArea::Min,
            |_, _event, window, _cx| window.minimize_window(),
            cx,
        );
        let maximize = self.render_titlebar_control(
            0x28c840,
            WindowControlArea::Max,
            |_, _event, window, _cx| window.zoom_window(),
            cx,
        );
        let controls = div()
            .h_full()
            .gap(px(8.))
            .items_center()
            .flex()
            .flex_none()
            .child(close)
            .child(minimize)
            .child(maximize);
        let sidebar_toggle = div()
            .id("sidebar-toggle")
            .size(px(28.))
            .items_center()
            .justify_center()
            .flex()
            .flex_none()
            .cursor_pointer()
            .hover(|style| style.bg(rgb(theme.tab_add_background)))
            .text_color(rgb(theme.ui_foreground))
            .child("◧")
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(|this, _event: &MouseDownEvent, _window, cx| {
                    this.toggle_sidebar(cx);
                    cx.stop_propagation();
                }),
            );
        let right_sidebar_placeholder = div()
            .id("right-sidebar-placeholder")
            .size(px(28.))
            .items_center()
            .justify_center()
            .flex()
            .flex_none()
            .opacity(0.45)
            .text_color(rgb(theme.ui_foreground))
            .child("◨")
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(|_this, _event: &MouseDownEvent, _window, cx| {
                    // Reserve this hitbox without allowing the titlebar drag
                    // handler to interpret the future sidebar control.
                    cx.stop_propagation();
                }),
            );
        div()
            .id("water-titlebar")
            .h(px(self.config.ui.titlebar_height))
            .w_full()
            .gap(px(8.))
            .px(px(10.))
            .items_center()
            .flex()
            // The titlebar owns its drag gesture explicitly below. Only the
            // three control hitboxes use WindowControlArea so they are not
            // shadowed by a full-width Drag hitbox.
            .bg(rgb(theme.chrome_background))
            .text_color(rgb(theme.ui_foreground))
            .on_mouse_down_out(cx.listener(|this, _event, _window, _cx| {
                this.titlebar_dragging = false;
            }))
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(|this, event: &MouseDownEvent, window, _cx| {
                    if event.click_count >= 2 {
                        // Double-click on the titlebar toggles zoom (the same
                        // native action as the maximize control); cancel the
                        // pending drag gesture so the second press cannot
                        // start a window move mid-zoom.
                        this.titlebar_dragging = false;
                        window.zoom_window();
                        return;
                    }
                    this.titlebar_dragging = true;
                }),
            )
            .on_mouse_up(
                MouseButton::Left,
                cx.listener(|this, event: &MouseUpEvent, _window, _cx| {
                    if event.click_count >= 2 {
                        return;
                    }
                    this.titlebar_dragging = false;
                }),
            )
            .on_mouse_move(cx.listener(|this, _event: &MouseMoveEvent, window, _cx| {
                if this.titlebar_dragging {
                    this.titlebar_dragging = false;
                    window.start_window_move();
                }
            }))
            .child(controls)
            .child(sidebar_toggle)
            .child(self.render_tab_bar(theme, cx))
            .child(right_sidebar_placeholder)
            .into_any_element()
    }

    fn render_pane_tree(
        &mut self,
        tab_id: TabId,
        tree: &PaneTreeDump,
        window_active: bool,
        metrics: TerminalMetrics,
        theme: ThemeColors,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let view = cx.entity();
        self.render_pane_tree_with_grow(
            tab_id,
            tree,
            &[],
            1.0,
            window_active,
            metrics,
            theme,
            view,
            cx,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn render_pane_tree_with_grow(
        &mut self,
        tab_id: TabId,
        tree: &PaneTreeDump,
        path: &[bool],
        grow: f32,
        window_active: bool,
        metrics: TerminalMetrics,
        theme: ThemeColors,
        view: Entity<WorkspaceView>,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        match tree {
            PaneTreeDump::Leaf {
                pane_id,
                surface_kind,
                surface_state,
                terminal,
                ..
            } => {
                let pane_id = *pane_id;
                let active = self.focused_pane == Some(pane_id);
                let border = if active {
                    rgb(theme.active_pane_border)
                } else {
                    rgb(theme.inactive_pane_border)
                };
                let label = match surface_kind {
                    crate::surface::SurfaceKind::Empty => "EmptySurface",
                    crate::surface::SurfaceKind::Terminal => "TerminalSurface",
                    crate::surface::SurfaceKind::Agent => "AgentSurface",
                    crate::surface::SurfaceKind::FileBrowser => "FileBrowserSurface",
                    crate::surface::SurfaceKind::ImagePreview => "ImagePreviewSurface",
                    crate::surface::SurfaceKind::MarkdownPreview => "MarkdownPreviewSurface",
                    crate::surface::SurfaceKind::Diff => "DiffSurface",
                };
                let terminal_id = terminal
                    .as_ref()
                    .map(|projection| projection.summary.terminal_id);
                let terminal_grid = terminal
                    .as_ref()
                    .map(|projection| {
                        self.local_terminal_snapshot(projection.summary.terminal_id, cx)
                    })
                    .flatten();
                let mouse_modes = terminal_grid
                    .as_ref()
                    .map(|snapshot| snapshot.modes)
                    .unwrap_or_default();
                let pi_agent_running = self.pi_agent_running_for_pane(pane_id);
                let smooth_scroll = !(mouse_modes.mouse_reporting && self.config.features.mouse_reporting)
                    && !(mouse_modes.alternate_screen && mouse_modes.alternate_scroll)
                    && !pi_agent_running;
                let content = if *surface_kind == crate::surface::SurfaceKind::Terminal {
                    terminal_grid
                        .map(|snapshot| {
                            crate::metrics::inc(crate::metrics::terminal_renders());
                            let ime_text = self.ime_marked_text_for(snapshot.terminal_id);
                            let scroll_offset_rows = if smooth_scroll {
                                self.terminal_scroll_offset_for_snapshot(&snapshot)
                            } else {
                                0.0
                            };
                            render_terminal_snapshot(
                                snapshot,
                                self.selection,
                                TerminalRenderOptions {
                                    metrics,
                                    theme,
                                    cursor_focused: active && window_active,
                                    scroll_offset_rows,
                                },
                                &self.config.terminal.font_family,
                                self.config.terminal.font_size,
                                ime_text,
                                self.terminal_bounds.clone(),
                                self.render_caches.clone(),
                                active.then(|| (view.clone(), self.focus_handle.clone())),
                            )
                        })
                        .unwrap_or_else(|| {
                            div()
                                .text_color(rgb(theme.terminal_foreground))
                                .child("Starting terminal…")
                                .into_any_element()
                        })
                } else {
                    div()
                        .text_color(rgb(theme.terminal_foreground))
                        .child(SharedString::from(format!("Pane {pane_id} · {label}")))
                        .into_any_element()
                };
                let content = match surface_state {
                    SurfaceState::Terminal(terminal) => div()
                        .size_full()
                        .min_w(px(0.))
                        .min_h(px(0.))
                        .overflow_hidden()
                        .relative()
                        .child(content)
                        .child(self.terminal_resize_observer(
                            self.active_connection,
                            pane_id,
                            terminal_id.expect("terminal surface carries a projection"),
                            TerminalSize::new(terminal.columns, terminal.lines),
                            metrics,
                            window_active,
                        ))
                        .into_any_element(),
                    SurfaceState::Empty(_) => content,
                };
                div()
                    .flex_1()
                    .flex_grow(grow)
                    .min_w(px(0.))
                    .min_h(px(0.))
                    .overflow_hidden()
                    .m(px(self.config.ui.pane_margin))
                    .p(px(self.config.ui.pane_padding))
                    .border_1()
                    .border_color(border)
                    .rounded(px(12.))
                    .bg(rgb(theme.pane_background))
                    .text_color(rgb(theme.terminal_foreground))
                    .child(content)
                    .on_mouse_down(
                        MouseButton::Left,
                        cx.listener(move |this, event: &MouseDownEvent, window, cx| {
                            this.focus_handle.focus(window, cx);
                            this.focused_pane = Some(pane_id);
                            // Resolve the terminal through the pane's own
                            // connection: pane IDs are not unique across
                            // local/remote servers.
                            let terminal_id =
                                this.terminal_id_for_pane_in_active_connection(pane_id);
                            if let Some(terminal_id) = terminal_id {
                                if this.begin_reported_mouse(terminal_id, event, cx) {
                                    cx.stop_propagation();
                                } else {
                                    this.begin_terminal_selection(
                                        terminal_id,
                                        event.position,
                                        event.modifiers.shift,
                                        cx,
                                    );
                                }
                            } else {
                                this.selection = None;
                            }
                            this.dispatch(
                                AppCommand::Pane(PaneCommand::Focus {
                                    pane_id: Some(pane_id),
                                    direction: None,
                                }),
                                cx,
                            );
                            // A terminal/pane click is an interaction, never
                            // a request to drag the native window.
                            cx.stop_propagation();
                        }),
                    )
                    .on_scroll_wheel(cx.listener(
                        move |this, event: &ScrollWheelEvent, window, cx| {
                            this.focus_handle.focus(window, cx);
                            let Some(terminal_id) = this.terminal_id_for_pane(pane_id) else {
                                return;
                            };
                            scroll_stat_inc(&SCROLL_WHEEL_EVENTS);
                            let input_kind = terminal_scroll_input_kind(
                                event,
                                this.active_trackpad_scrolls.contains(&terminal_id),
                            );
                            match event.touch_phase {
                                TouchPhase::Started
                                    if matches!(event.delta, ScrollDelta::Pixels(_)) =>
                                {
                                    this.active_trackpad_scrolls.insert(terminal_id);
                                }
                                TouchPhase::Ended | TouchPhase::Cancelled => {
                                    this.active_trackpad_scrolls.remove(&terminal_id);
                                }
                                _ => {}
                            }
                            scroll_stat_inc(match input_kind {
                                TerminalScrollInputKind::TrackpadGesture => &SCROLL_TRACKPAD_EVENTS,
                                TerminalScrollInputKind::MouseWheel => &SCROLL_MOUSE_EVENTS,
                            });
                            let delta_rows =
                                terminal_scroll_delta_rows(event, this.terminal_metrics);
                            if !delta_rows.is_finite() || delta_rows == 0.0 {
                                return;
                            }

                            let should_repaint;
                            if mouse_modes.mouse_reporting && this.config.features.mouse_reporting {
                                // Mouse reporting owns the wheel protocol; do
                                // not turn a partial trackpad delta into local
                                // viewport movement in this mode.
                                should_repaint =
                                    this.scroll_accumulators.remove(&terminal_id).is_some();
                                this.mouse_scroll_animations.remove(&terminal_id);
                                if let Some(bytes) = terminal_mouse_input(
                                    event,
                                    TerminalMouseContext {
                                        modes: mouse_modes,
                                        bounds: this.terminal_bounds_for(terminal_id),
                                        metrics: this.terminal_metrics,
                                    },
                                ) {
                                    this.enqueue_terminal_command(
                                        terminal_id,
                                        TerminalCommand::SendBytes {
                                            terminal_id: Some(terminal_id),
                                            pane_id: Some(pane_id),
                                            bytes,
                                        },
                                    );
                                }
                            } else if mouse_modes.alternate_screen && mouse_modes.alternate_scroll {
                                // Alternate-screen applications expect cursor
                                // key sequences rather than normal scrollback.
                                should_repaint =
                                    this.scroll_accumulators.remove(&terminal_id).is_some();
                                this.mouse_scroll_animations.remove(&terminal_id);
                                let lines = terminal_scroll_lines(event, this.terminal_metrics);
                                if lines != 0 {
                                    this.enqueue_terminal_command(
                                        terminal_id,
                                        TerminalCommand::SendText {
                                            terminal_id: Some(terminal_id),
                                            pane_id: Some(pane_id),
                                            text: terminal_alternate_scroll_input(
                                                lines,
                                                mouse_modes,
                                            ),
                                        },
                                    );
                                }
                            } else {
                                if input_kind == TerminalScrollInputKind::MouseWheel
                                    && delta_rows.abs() > 1.0
                                {
                                    should_repaint = this.begin_mouse_scroll_animation(
                                        terminal_id,
                                        pane_id,
                                        delta_rows,
                                        window,
                                        cx,
                                    );
                                } else {
                                    // Trackpad motion (including system
                                    // inertia) stays one-to-one with AppKit's
                                    // continuous pixel delta. Never layer an
                                    // extra easing curve over it.
                                    this.mouse_scroll_animations.remove(&terminal_id);
                                    let (target, repaint) =
                                        this.accumulate_terminal_scroll(terminal_id, delta_rows);
                                    should_repaint = repaint;
                                    if let Some(target) = target {
                                        this.queue_terminal_viewport_request(
                                            terminal_id,
                                            pane_id,
                                            target,
                                            window,
                                            cx,
                                        );
                                    }
                                }
                            }
                            // GPUI collapses invalidations within a frame. Do
                            // notify for every effective fractional delta so
                            // visual motion can never wait on another event
                            // or on the worker acknowledgement.
                            if should_repaint {
                                cx.notify();
                            }
                            // Consume even a fractional normal-screen wheel
                            // event so the surrounding UI cannot interpret it
                            // as a second scroll gesture.
                            cx.stop_propagation();
                        },
                    ))
                    .into_any_element()
            }
            PaneTreeDump::Split {
                axis,
                ratio,
                first,
                second,
            } => {
                let split_axis = *axis;
                let preview_ratio = self
                    .split_drag
                    .as_ref()
                    .filter(|drag| {
                        drag.tab_id == tab_id && drag.path == path && drag.axis == split_axis
                    })
                    .and_then(|drag| drag.preview_ratio)
                    .unwrap_or(*ratio)
                    .clamp(0.05, 0.95);
                let mut first_path = path.to_vec();
                first_path.push(false);
                let mut second_path = path.to_vec();
                second_path.push(true);
                let divider_path = path.to_vec();
                let divider_cursor = if split_axis == SplitAxis::Horizontal {
                    CursorStyle::ResizeLeftRight
                } else {
                    CursorStyle::ResizeUpDown
                };
                let divider = {
                    let divider_width = SPLIT_DIVIDER_WIDTH_PX;
                    let divider_id = path
                        .iter()
                        .map(|bit| if *bit { '1' } else { '0' })
                        .collect::<String>();
                    let divider = div()
                        .id(format!("pane-divider-{tab_id}-{divider_id}"))
                        .flex_none()
                        .cursor(divider_cursor)
                        .bg(rgba(0x00000000))
                        .hover(|style| style.bg(rgb(theme.inactive_pane_border)))
                        .on_mouse_down(
                            MouseButton::Left,
                            cx.listener(move |this, event: &MouseDownEvent, window, cx| {
                                this.focus_handle.focus(window, cx);
                                this.begin_split_drag(
                                    tab_id,
                                    divider_path.clone(),
                                    split_axis,
                                    event.position,
                                    cx,
                                );
                                cx.stop_propagation();
                            }),
                        );
                    if split_axis == SplitAxis::Horizontal {
                        divider.w(px(divider_width)).h_full()
                    } else {
                        divider.h(px(divider_width)).w_full()
                    }
                };
                let split_bounds = self.split_bounds.clone();
                let split_path = path.to_vec();
                let bounds_observer = canvas(
                    move |bounds, _, _| {
                        let extent = if split_axis == SplitAxis::Horizontal {
                            f32::from(bounds.size.width)
                        } else {
                            f32::from(bounds.size.height)
                        };
                        let origin = if split_axis == SplitAxis::Horizontal {
                            f32::from(bounds.origin.x)
                        } else {
                            f32::from(bounds.origin.y)
                        };
                        split_bounds
                            .lock()
                            .expect("split bounds poisoned")
                            .insert((tab_id, split_path), SplitRect { origin, extent });
                    },
                    |_bounds, _, _, _| {},
                )
                .absolute()
                .inset_0();
                let mut container = div()
                    .flex_1()
                    .flex_grow(grow)
                    .flex()
                    .min_w(px(0.))
                    .min_h(px(0.))
                    .overflow_hidden();
                if split_axis == SplitAxis::Horizontal {
                    container = container.flex_row();
                } else {
                    container = container.flex_col();
                }
                container
                    .child(self.render_pane_tree_with_grow(
                        tab_id,
                        first,
                        &first_path,
                        preview_ratio,
                        window_active,
                        metrics,
                        theme,
                        view.clone(),
                        cx,
                    ))
                    .child(divider)
                    .child(self.render_pane_tree_with_grow(
                        tab_id,
                        second,
                        &second_path,
                        1.0 - preview_ratio,
                        window_active,
                        metrics,
                        theme,
                        view,
                        cx,
                    ))
                    .child(bounds_observer)
                    .into_any_element()
            }
        }
    }

    fn render_active_tab(
        &mut self,
        window_active: bool,
        metrics: TerminalMetrics,
        theme: ThemeColors,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let Some(workspace) = self.selected_workspace_dump() else {
            return render_empty_workspace_state(theme, &self.config.shortcuts.new_workspace);
        };
        let Some(active_tab_id) = workspace.active_tab else {
            return render_empty_tab_state(theme, &self.config.shortcuts.new_terminal_tab);
        };
        let Some(tab) = workspace.tabs.iter().find(|tab| tab.id == active_tab_id) else {
            return render_empty_tab_state(theme, &self.config.shortcuts.new_terminal_tab);
        };
        let tab_id = tab.id;
        let tree = tab.tree.clone();
        self.render_pane_tree(tab_id, &tree, window_active, metrics, theme, cx)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct SidebarGroupGeometry {
    top: f32,
    bottom: f32,
}

/// Finds the nearest visible boundary between workspace groups. The returned
/// index is an insertion slot in the original displayed order; callers map it
/// to the command's after-removal index before dispatching.
fn sidebar_drop_boundary(
    pointer_y: f32,
    groups: &[SidebarGroupGeometry],
    viewport_top: f32,
    viewport_bottom: f32,
    tolerance: f32,
) -> Option<(usize, f32)> {
    if groups.is_empty()
        || !pointer_y.is_finite()
        || !viewport_top.is_finite()
        || !viewport_bottom.is_finite()
        || pointer_y < viewport_top
        || pointer_y > viewport_bottom
    {
        return None;
    }
    let tolerance = tolerance.max(0.0);
    let mut nearest: Option<(usize, f32)> = None;
    for (index, y) in std::iter::once((0, groups[0].top)).chain(
        groups
            .iter()
            .enumerate()
            .map(|(index, group)| (index + 1, group.bottom)),
    ) {
        if !y.is_finite() || y < viewport_top - tolerance || y > viewport_bottom + tolerance {
            continue;
        }
        let distance = (pointer_y - y).abs();
        if nearest.is_none_or(|(_, nearest_y)| distance < (pointer_y - nearest_y).abs()) {
            nearest = Some((index, y));
        }
    }
    nearest.filter(|(_, y)| (pointer_y - *y).abs() <= tolerance)
}

/// Maps an original-order insertion slot to the final index after removing
/// the dragged workspace. The command dispatcher uses this convention.
fn sidebar_reorder_final_index(source_index: usize, boundary_index: usize, count: usize) -> usize {
    if count == 0 || source_index >= count {
        return 0;
    }
    let boundary_index = boundary_index.min(count);
    let index = if boundary_index <= source_index {
        boundary_index
    } else {
        boundary_index.saturating_sub(1)
    };
    index.min(count.saturating_sub(1))
}

fn split_pointer_coordinate(position: Point<gpui::Pixels>, axis: SplitAxis) -> f32 {
    match axis {
        SplitAxis::Horizontal => f32::from(position.x),
        SplitAxis::Vertical => f32::from(position.y),
    }
}

fn split_ratio_for_pointer(pointer: f32, origin: f32, extent: f32, divider: f32) -> f32 {
    if !pointer.is_finite() || !origin.is_finite() || !extent.is_finite() {
        return 0.5;
    }
    // The divider is a fixed-size flex item: children share (extent -
    // divider). The divider CENTER sits at origin + ratio * usable +
    // divider / 2, which inverts to the formula below so the committed
    // ratio matches what the pointer points at.
    let usable = extent - divider.max(0.0);
    if usable <= 0.0 {
        return 0.5;
    }
    (((pointer - origin) - divider / 2.0) / usable).clamp(0.05, 0.95)
}

fn pane_split_at_path<'a>(
    tree: &'a PaneTreeDump,
    path: &[bool],
) -> Option<(SplitAxis, f32, &'a PaneTreeDump, &'a PaneTreeDump)> {
    if let Some((head, tail)) = path.split_first() {
        match tree {
            PaneTreeDump::Split { first, second, .. } => {
                pane_split_at_path(if *head { second } else { first }, tail)
            }
            PaneTreeDump::Leaf { .. } => None,
        }
    } else {
        match tree {
            PaneTreeDump::Split {
                axis,
                ratio,
                first,
                second,
            } => Some((*axis, *ratio, first, second)),
            PaneTreeDump::Leaf { .. } => None,
        }
    }
}

fn workspace_exists_in_snapshot(snapshot: &ModelSnapshot, workspace_id: WorkspaceId) -> bool {
    if snapshot.workspaces.is_empty() {
        snapshot
            .workspace
            .as_ref()
            .is_some_and(|workspace| workspace.id == workspace_id)
    } else {
        snapshot
            .workspaces
            .iter()
            .any(|workspace| workspace.id == workspace_id)
    }
}

fn first_workspace_id_in_snapshot(snapshot: &ModelSnapshot) -> Option<WorkspaceId> {
    if snapshot.workspaces.is_empty() {
        snapshot.workspace.as_ref().map(|workspace| workspace.id)
    } else {
        snapshot.workspaces.first().map(|workspace| workspace.id)
    }
}

fn workspace_dump_for_snapshot(
    snapshot: &ModelSnapshot,
    workspace_id: WorkspaceId,
) -> Option<&WorkspaceDump> {
    if snapshot.workspaces.is_empty() {
        snapshot
            .workspace
            .as_ref()
            .filter(|workspace| workspace.id == workspace_id)
    } else {
        snapshot
            .workspaces
            .iter()
            .find(|workspace| workspace.id == workspace_id)
    }
}

fn workspace_selection_after_snapshot(
    selected_workspace: Option<WorkspaceId>,
    snapshot: &ModelSnapshot,
) -> Option<WorkspaceId> {
    selected_workspace
        .filter(|workspace_id| workspace_exists_in_snapshot(snapshot, *workspace_id))
        .or_else(|| {
            snapshot
                .active_workspace
                .filter(|workspace_id| workspace_exists_in_snapshot(snapshot, *workspace_id))
        })
        .or_else(|| first_workspace_id_in_snapshot(snapshot))
}

fn workspace_active_pane(workspace: &WorkspaceDump) -> Option<PaneId> {
    workspace
        .active_tab
        .and_then(|tab_id| workspace.tabs.iter().find(|tab| tab.id == tab_id))
        .map(|tab| tab.active_pane)
}

fn pane_tree_contains_pane(tree: &PaneTreeDump, pane_id: PaneId) -> bool {
    match tree {
        PaneTreeDump::Leaf {
            pane_id: leaf_id, ..
        } => *leaf_id == pane_id,
        PaneTreeDump::Split { first, second, .. } => {
            pane_tree_contains_pane(first, pane_id) || pane_tree_contains_pane(second, pane_id)
        }
    }
}

fn workspace_active_tab_contains_pane(workspace: &WorkspaceDump, pane_id: PaneId) -> bool {
    workspace
        .active_tab
        .and_then(|tab_id| workspace.tabs.iter().find(|tab| tab.id == tab_id))
        .is_some_and(|tab| pane_tree_contains_pane(&tab.tree, pane_id))
}

fn focused_pane_for_workspace(
    snapshot: &ModelSnapshot,
    workspace_id: Option<WorkspaceId>,
    preferred_pane: Option<PaneId>,
) -> Option<PaneId> {
    let workspace_id = workspace_id?;
    let workspace = workspace_dump_for_snapshot(snapshot, workspace_id)?;
    preferred_pane
        .filter(|pane_id| workspace_active_tab_contains_pane(workspace, *pane_id))
        .or_else(|| {
            (snapshot.active_workspace == Some(workspace_id))
                .then_some(snapshot.focused_pane)
                .flatten()
                .filter(|pane_id| workspace_active_tab_contains_pane(workspace, *pane_id))
        })
        .or_else(|| workspace_active_pane(workspace))
}

/// Whether the terminal is projected anywhere in the state (grid or summary
/// only). Hidden tabs project summaries, so this is the right liveness test
/// for per-terminal UI bookkeeping.
fn terminal_projection_in_snapshot(
    snapshot: &ModelSnapshot,
    terminal_id: TerminalId,
) -> Option<&crate::app::model::TerminalProjection> {
    let tabs = if snapshot.workspaces.is_empty() {
        snapshot
            .workspace
            .as_ref()
            .into_iter()
            .flat_map(|workspace| workspace.tabs.iter())
            .collect::<Vec<_>>()
    } else {
        snapshot
            .workspaces
            .iter()
            .flat_map(|workspace| workspace.tabs.iter())
            .collect::<Vec<_>>()
    };
    tabs.into_iter()
        .find_map(|tab| terminal_projection_for_id(&tab.tree, terminal_id))
}

fn terminal_projection_for_id(
    tree: &PaneTreeDump,
    terminal_id: TerminalId,
) -> Option<&crate::app::model::TerminalProjection> {
    match tree {
        PaneTreeDump::Leaf {
            surface_state,
            terminal,
            ..
        } => match surface_state {
            SurfaceState::Terminal(terminal_state) if terminal_state.terminal_id == terminal_id => {
                terminal.as_deref()
            }
            SurfaceState::Terminal(_) | SurfaceState::Empty(_) => None,
        },
        PaneTreeDump::Split { first, second, .. } => terminal_projection_for_id(first, terminal_id)
            .or_else(|| terminal_projection_for_id(second, terminal_id)),
    }
}

fn render_empty_workspace_state(theme: ThemeColors, shortcut: &str) -> AnyElement {
    render_empty_state(
        theme,
        "No workspace",
        format!(
            "Press {} to create a workspace",
            shortcut_label(shortcut, "cmd-shift-n")
        ),
    )
}

fn render_empty_tab_state(theme: ThemeColors, shortcut: &str) -> AnyElement {
    render_empty_state(
        theme,
        "No terminal tab",
        format!(
            "Press {} to open a terminal tab",
            shortcut_label(shortcut, "cmd-t")
        ),
    )
}

fn render_empty_state(theme: ThemeColors, title: &'static str, hint: String) -> AnyElement {
    div()
        .flex_1()
        .items_center()
        .justify_center()
        .flex()
        .flex_col()
        .gap(px(8.))
        .text_color(rgb(theme.terminal_foreground))
        .child(title)
        .child(
            div()
                .text_color(rgb(theme.inactive_pane_border))
                .child(SharedString::from(hint)),
        )
        .into_any_element()
}

fn shortcut_label(source: &str, fallback: &str) -> String {
    let source = source.trim();
    let source = if !source.is_empty()
        && source
            .split_whitespace()
            .all(|keystroke| Keystroke::parse(keystroke).is_ok())
    {
        source.split_whitespace().next().unwrap_or(fallback)
    } else {
        fallback
    };
    source
        .split('-')
        .map(|part| match part.to_ascii_lowercase().as_str() {
            "cmd" => "Cmd".to_owned(),
            "ctrl" | "control" => "Ctrl".to_owned(),
            "shift" => "Shift".to_owned(),
            "alt" | "option" => "Alt".to_owned(),
            "fn" | "function" => "Fn".to_owned(),
            part => {
                let mut characters = part.chars();
                match characters.next() {
                    Some(first) => first.to_uppercase().collect::<String>() + characters.as_str(),
                    None => String::new(),
                }
            }
        })
        .collect::<Vec<_>>()
        .join("-")
}

impl Focusable for WorkspaceView {
    fn focus_handle(&self, _cx: &App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

impl EntityInputHandler for WorkspaceView {
    fn text_for_range(
        &mut self,
        range_utf16: Range<usize>,
        adjusted_range: &mut Option<Range<usize>>,
        _window: &mut Window,
        _cx: &mut Context<Self>,
    ) -> Option<String> {
        let range = normalize_utf16_range(range_utf16, utf16_len(&self.ime_marked_text));
        let start = utf16_offset_to_byte(&self.ime_marked_text, range.start);
        let end = utf16_offset_to_byte_end(&self.ime_marked_text, range.end);
        adjusted_range.replace(range);
        Some(self.ime_marked_text[start..end].to_owned())
    }

    fn selected_text_range(
        &mut self,
        _ignore_disabled_input: bool,
        _window: &mut Window,
        _cx: &mut Context<Self>,
    ) -> Option<UTF16Selection> {
        if self.active_terminal_id().is_none() && self.ime_terminal.is_none() {
            return None;
        }
        let range = normalize_utf16_range(
            self.ime_selected_range.clone(),
            utf16_len(&self.ime_marked_text),
        );
        Some(UTF16Selection {
            range: range.clone(),
            reversed: false,
        })
    }

    fn marked_text_range(
        &self,
        _window: &mut Window,
        _cx: &mut Context<Self>,
    ) -> Option<Range<usize>> {
        (!self.ime_marked_text.is_empty() && self.ime_terminal.is_some())
            .then(|| 0..utf16_len(&self.ime_marked_text))
    }

    fn unmark_text(&mut self, _window: &mut Window, cx: &mut Context<Self>) {
        if self.dialog_is_text_input() {
            let base = self.dialog_ime_base.clamp(0, self.dialog_caret);
            self.dialog_input.replace_range(base..self.dialog_caret, "");
            self.dialog_caret = base;
        }
        self.reset_ime_marked_text();
        cx.notify();
    }

    fn replace_text_in_range(
        &mut self,
        _range_utf16: Option<Range<usize>>,
        new_text: &str,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if self.dialog_is_text_input() {
            // The IME commits through this hook; land the composed text at
            // the dialog caret instead of the terminal below the scrim.
            let base = self.dialog_ime_base.clamp(0, self.dialog_caret);
            self.dialog_input.replace_range(base..self.dialog_caret, "");
            if !new_text.is_empty() {
                self.dialog_input.insert_str(base, new_text);
            }
            self.dialog_caret = base + new_text.len();
            self.dialog_ime_base = self.dialog_caret;
            self.dialog_caret_visible = true;
            self.remote_connection_error = None;
            self.reset_ime_marked_text();
            cx.notify();
            return;
        }
        let Some(terminal_id) = self.ime_terminal.or_else(|| self.active_terminal_id()) else {
            return;
        };
        self.ime_terminal = Some(terminal_id);
        self.selection = None;
        self.reset_ime_marked_text();
        if !new_text.is_empty() {
            self.focus_terminal_live_bottom(terminal_id, cx);
            self.enqueue_terminal_command(
                terminal_id,
                TerminalCommand::SendText {
                    terminal_id: Some(terminal_id),
                    pane_id: None,
                    text: new_text.to_owned(),
                },
            );
        }
        cx.notify();
    }

    fn replace_and_mark_text_in_range(
        &mut self,
        range_utf16: Option<Range<usize>>,
        new_text: &str,
        new_selected_range: Option<Range<usize>>,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if self.dialog_is_text_input() {
            // Keep the in-progress composition at the dialog caret so pinyin
            // (or any marked-text IME) can rename workspaces directly.
            let base = self.dialog_ime_base.clamp(0, self.dialog_caret);
            self.dialog_input.replace_range(base..self.dialog_caret, "");
            self.dialog_input.insert_str(base, new_text);
            self.dialog_caret = base + new_text.len();
            self.dialog_caret_visible = true;
            self.ime_terminal = self.active_terminal_id();
            self.ime_marked_text = new_text.to_owned();
            self.ime_selected_range = 0..utf16_len(new_text);
            cx.notify();
            return;
        }
        let Some(terminal_id) = self.ime_terminal.or_else(|| self.active_terminal_id()) else {
            return;
        };
        self.ime_terminal = Some(terminal_id);
        let current = self.ime_marked_text.clone();
        let replacement = range_utf16
            .map(|range| normalize_utf16_range(range, utf16_len(&current)))
            .unwrap_or_else(|| 0..utf16_len(&current));
        let start = utf16_offset_to_byte(&current, replacement.start);
        let end = utf16_offset_to_byte_end(&current, replacement.end);
        let mut updated = String::with_capacity(
            current.len().saturating_sub(end.saturating_sub(start)) + new_text.len(),
        );
        updated.push_str(&current[..start]);
        updated.push_str(new_text);
        updated.push_str(&current[end..]);
        self.ime_marked_text = updated;
        let new_length = utf16_len(new_text);
        let replacement_start = replacement.start;
        self.ime_selected_range = new_selected_range
            .map(|range| {
                let range = normalize_utf16_range(range, new_length);
                replacement_start.saturating_add(range.start)
                    ..replacement_start.saturating_add(range.end)
            })
            .unwrap_or_else(|| {
                let end = replacement_start.saturating_add(new_length);
                end..end
            });
        if !new_text.is_empty() {
            self.focus_terminal_live_bottom(terminal_id, cx);
        } else {
            self.selection = None;
        }
        cx.notify();
    }

    fn bounds_for_range(
        &mut self,
        _range_utf16: Range<usize>,
        element_bounds: Bounds<gpui::Pixels>,
        _window: &mut Window,
        _cx: &mut Context<Self>,
    ) -> Option<Bounds<gpui::Pixels>> {
        let terminal_id = self.ime_terminal.or_else(|| self.active_terminal_id())?;
        let snapshot = self.terminal_snapshot_for(terminal_id);
        let (cursor, width_columns) = snapshot
            .map(|snapshot| {
                let cursor = terminal_cursor_position(snapshot);
                let width_columns = snapshot
                    .cell(cursor.0, cursor.1)
                    .map(|cell| if cell.flags.wide() { 2 } else { 1 })
                    .unwrap_or(1);
                (cursor, width_columns)
            })
            .unwrap_or(((0, 0), 1));
        Some(terminal_cell_bounds(
            element_bounds,
            self.terminal_metrics,
            cursor.0,
            cursor.1,
            width_columns,
        ))
    }

    fn character_index_for_point(
        &mut self,
        point: Point<gpui::Pixels>,
        _window: &mut Window,
        _cx: &mut Context<Self>,
    ) -> Option<usize> {
        let terminal_id = self.ime_terminal.or_else(|| self.active_terminal_id())?;
        let bounds = self.terminal_bounds_for(terminal_id)?;
        let mouse = terminal_mouse_position(point, Some(bounds), self.terminal_metrics);
        let row = mouse.row.saturating_sub(1);
        let column = mouse.column.saturating_sub(1);
        let (cursor_row, cursor_column) = self
            .terminal_snapshot_for(terminal_id)
            .map(terminal_cursor_position)
            .unwrap_or((0, 0));
        let offset = if row == cursor_row && column >= cursor_column {
            utf16_len(&self.ime_marked_text).min(column - cursor_column)
        } else {
            0
        };
        Some(offset)
    }

    fn set_selected_text_range(
        &mut self,
        range_utf16: Range<usize>,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.ime_selected_range =
            normalize_utf16_range(range_utf16, utf16_len(&self.ime_marked_text));
        cx.notify();
    }

    fn text_length_utf16(
        &mut self,
        _window: &mut Window,
        _cx: &mut Context<Self>,
    ) -> Option<usize> {
        self.active_terminal_id()
            .or(self.ime_terminal)
            .map(|_| utf16_len(&self.ime_marked_text))
    }

    fn accepts_text_input(&self, _window: &mut Window, _cx: &mut Context<Self>) -> bool {
        self.active_terminal_id().is_some() || self.ime_terminal.is_some()
    }

    fn text_input_configuration(
        &mut self,
        _window: &mut Window,
        _cx: &mut Context<Self>,
    ) -> TextInputConfiguration {
        TextInputConfiguration::default()
    }

    fn text_input_editable_range(
        &mut self,
        _window: &mut Window,
        _cx: &mut Context<Self>,
    ) -> Option<Range<usize>> {
        self.active_terminal_id()
            .or(self.ime_terminal)
            .map(|_| 0..utf16_len(&self.ime_marked_text))
    }
}

fn utf16_len(text: &str) -> usize {
    text.encode_utf16().count()
}

fn utf16_offset_to_byte(text: &str, offset: usize) -> usize {
    let mut current = 0;
    for (byte, character) in text.char_indices() {
        if offset <= current {
            return byte;
        }
        let next = current + character.len_utf16();
        if offset < next {
            return byte;
        }
        current = next;
        if offset == current {
            return byte + character.len_utf8();
        }
    }
    text.len()
}

fn utf16_offset_to_byte_end(text: &str, offset: usize) -> usize {
    let mut current = 0;
    for (byte, character) in text.char_indices() {
        let next = current + character.len_utf16();
        if offset <= current {
            return byte;
        }
        if offset <= next {
            return byte + character.len_utf8();
        }
        current = next;
    }
    text.len()
}

fn normalize_utf16_range(range: Range<usize>, length: usize) -> Range<usize> {
    let start = range.start.min(length);
    let end = range.end.min(length);
    if start <= end { start..end } else { end..end }
}

/// Drifts a terminal selection to follow a viewport move applied locally.
fn shift_selection_for_viewport(
    selection: &mut Option<TerminalSelection>,
    terminal_id: TerminalId,
    previous: &TerminalSnapshot,
    next: &TerminalSnapshot,
) {
    let Some(selection_state) = selection
        .as_mut()
        .filter(|selection| selection.terminal_id == terminal_id)
    else {
        return;
    };
    let delta = next
        .viewport_position
        .saturating_sub(previous.viewport_position);
    shift_terminal_selection_rows(selection_state, delta);
}

/// Rebases all UI-local viewport state as soon as the local emulator applies
/// a requested position. There is no worker acknowledgement in the raw-stream
/// architecture: leaving the accumulated offset pending would apply the same
/// scroll twice and cap subsequent gestures at the accumulator's safety bound.
fn reconcile_local_viewport_snapshot(
    selection: &mut Option<TerminalSelection>,
    scroll_accumulators: &mut BTreeMap<TerminalId, TerminalScrollState>,
    terminal_id: TerminalId,
    previous: &TerminalSnapshot,
    next: &TerminalSnapshot,
) {
    shift_selection_for_viewport(selection, terminal_id, previous, next);
    if let Some(state) = scroll_accumulators.get_mut(&terminal_id) {
        reconcile_visual_scroll(state, next);
    }
}

fn shift_terminal_selection_rows(selection: &mut TerminalSelection, delta: i64) {
    let delta = delta.clamp(i32::MIN as i64, i32::MAX as i64) as i32;
    selection.anchor.position.row = selection.anchor.position.row.saturating_add(delta);
    selection.head.position.row = selection.head.position.row.saturating_add(delta);
}

fn terminal_id_for_pane(tree: &PaneTreeDump, pane_id: PaneId) -> Option<TerminalId> {
    match tree {
        PaneTreeDump::Leaf {
            pane_id: leaf_id,
            surface_state,
            ..
        } if *leaf_id == pane_id => match surface_state {
            SurfaceState::Terminal(terminal) => Some(terminal.terminal_id),
            SurfaceState::Empty(_) => None,
        },
        PaneTreeDump::Leaf { .. } => None,
        PaneTreeDump::Split { first, second, .. } => {
            terminal_id_for_pane(first, pane_id).or_else(|| terminal_id_for_pane(second, pane_id))
        }
    }
}

fn collect_terminal_ids(tree: &PaneTreeDump, terminal_ids: &mut BTreeSet<TerminalId>) {
    match tree {
        PaneTreeDump::Leaf { terminal, .. } => {
            if let Some(terminal) = terminal {
                terminal_ids.insert(terminal.summary.terminal_id);
            }
        }
        PaneTreeDump::Split { first, second, .. } => {
            collect_terminal_ids(first, terminal_ids);
            collect_terminal_ids(second, terminal_ids);
        }
    }
}

/// Registers window-level listeners during paint so terminal selection and
/// view-local drag gestures keep receiving moves and release events after the
/// pointer crosses another pane or the root hitbox.
fn workspace_mouse_event_observer(entity: Entity<WorkspaceView>) -> AnyElement {
    canvas(
        |_bounds, _, _| {},
        move |_bounds, _, window, _| {
            let move_entity = entity.clone();
            window.on_mouse_event(move |event: &MouseMoveEvent, phase, window, cx| {
                if phase != DispatchPhase::Capture {
                    return;
                }
                let (handled, start_window_move) = move_entity.update(cx, |view, cx| {
                    if view.dragging_sidebar {
                        view.update_sidebar_width(event.position.x, cx);
                        (true, false)
                    } else if view.sidebar_drag.is_some() {
                        (view.update_sidebar_drag(event.position, cx), false)
                    } else if view.split_drag.is_some() {
                        (view.update_split_drag(event.position, cx), false)
                    } else if view.window_drag_start.is_some() {
                        let start_window_move = event.pressed_button == Some(MouseButton::Left)
                            && view.update_window_drag(event.position);
                        // Keep a pending background gesture from being
                        // interpreted as a terminal click if the pointer
                        // crosses into a pane before the threshold.
                        (true, start_window_move)
                    } else {
                        view.update_terminal_selection(event, window, cx);
                        (false, false)
                    }
                });
                if start_window_move {
                    window.start_window_move();
                }
                if handled {
                    cx.stop_propagation();
                }
            });

            let up_entity = entity;
            window.on_mouse_event(move |event: &MouseUpEvent, phase, _window, cx| {
                if phase != DispatchPhase::Capture || event.button != MouseButton::Left {
                    return;
                }
                let handled = up_entity.update(cx, |view, cx| {
                    view.window_drag_start = None;
                    if view.dragging_sidebar {
                        view.dragging_sidebar = false;
                        cx.notify();
                        true
                    } else if view.sidebar_drag.is_some() {
                        view.finish_sidebar_drag(cx);
                        true
                    } else if view.split_drag.is_some() {
                        view.finish_split_drag(cx);
                        true
                    } else {
                        view.finish_terminal_selection(event, cx);
                        false
                    }
                });
                if handled {
                    cx.stop_propagation();
                }
            });
        },
    )
    .size_full()
    .absolute()
    .inset_0()
    .into_any_element()
}

fn terminal_scroll_delta_rows(event: &ScrollWheelEvent, metrics: TerminalMetrics) -> f32 {
    match event.delta {
        ScrollDelta::Lines(delta) => delta.y,
        ScrollDelta::Pixels(delta) => f32::from(delta.y) / metrics.line_height.max(f32::EPSILON),
    }
}

fn terminal_selection_autoscroll_direction(y_in_pane: f32, pane_height: f32) -> Option<i64> {
    if pane_height <= 0.0 {
        return None;
    }
    if y_in_pane < TERMINAL_SELECTION_AUTOSCROLL_MARGIN_PX {
        Some(TERMINAL_SELECTION_AUTOSCROLL_STEP_ROWS)
    } else if y_in_pane > pane_height - TERMINAL_SELECTION_AUTOSCROLL_MARGIN_PX {
        Some(-TERMINAL_SELECTION_AUTOSCROLL_STEP_ROWS)
    } else {
        None
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TerminalScrollInputKind {
    TrackpadGesture,
    MouseWheel,
}

fn terminal_scroll_input_kind(
    event: &ScrollWheelEvent,
    trackpad_gesture_active: bool,
) -> TerminalScrollInputKind {
    match event.delta {
        ScrollDelta::Lines(_) => TerminalScrollInputKind::MouseWheel,
        ScrollDelta::Pixels(_) => match event.touch_phase {
            TouchPhase::Started | TouchPhase::Ended | TouchPhase::Cancelled => {
                TerminalScrollInputKind::TrackpadGesture
            }
            TouchPhase::Moved if trackpad_gesture_active => {
                TerminalScrollInputKind::TrackpadGesture
            }
            TouchPhase::Moved => TerminalScrollInputKind::MouseWheel,
        },
    }
}

fn terminal_scroll_lines(event: &ScrollWheelEvent, metrics: TerminalMetrics) -> i32 {
    let delta = terminal_scroll_delta_rows(event, metrics);
    if delta == 0.0 {
        return 0;
    }
    (if delta.abs() < 1.0 {
        delta.signum()
    } else {
        delta.round()
    } as i32)
        .clamp(-100, 100)
}

fn terminal_alternate_scroll_input(lines: i32, modes: TerminalModes) -> String {
    let final_character = if lines > 0 { 'A' } else { 'B' };
    let sequence = cursor_sequence(final_character, 1, modes.application_cursor);
    sequence.repeat(lines.unsigned_abs().min(100) as usize)
}

fn terminal_mouse_input(event: &ScrollWheelEvent, mouse: TerminalMouseContext) -> Option<Vec<u8>> {
    let lines = terminal_scroll_lines(event, mouse.metrics);
    if lines == 0 {
        return None;
    }
    let button = if lines > 0 { 64_u16 } else { 65_u16 };
    let count = lines.unsigned_abs().min(100) as usize;
    Some(terminal_mouse_sequence(
        button,
        count,
        event.position,
        event.modifiers,
        mouse,
        TerminalMouseReportKind::Press,
    ))
}

fn terminal_mouse_button_input(
    position: Point<gpui::Pixels>,
    button: MouseButton,
    pressed: bool,
    motion: bool,
    modifiers: gpui::Modifiers,
    mouse: TerminalMouseContext,
) -> Option<Vec<u8>> {
    let button = mouse_button_code(button)?;
    Some(terminal_mouse_sequence(
        button,
        1,
        position,
        modifiers,
        mouse,
        if pressed {
            if motion {
                TerminalMouseReportKind::Motion
            } else {
                TerminalMouseReportKind::Press
            }
        } else {
            TerminalMouseReportKind::Release
        },
    ))
}

fn terminal_mouse_sequence(
    button: u16,
    count: usize,
    position: Point<gpui::Pixels>,
    modifiers: gpui::Modifiers,
    mouse: TerminalMouseContext,
    kind: TerminalMouseReportKind,
) -> Vec<u8> {
    let release = kind == TerminalMouseReportKind::Release;
    let motion = kind == TerminalMouseReportKind::Motion;
    let modifier = u16::from(modifiers.shift) * 4
        + u16::from(modifiers.alt) * 8
        + u16::from(modifiers.control) * 16;
    let button = if release {
        3
    } else {
        button + if motion { 32 } else { 0 }
    } + modifier;
    let mouse_position = terminal_mouse_position(position, mouse.bounds, mouse.metrics);
    if mouse.modes.sgr_mouse {
        let suffix = if release { 'm' } else { 'M' };
        let mut input = Vec::with_capacity(count * 16);
        for _ in 0..count {
            input.extend_from_slice(
                format!(
                    "\u{1b}[<{button};{};{}{suffix}",
                    mouse_position.column, mouse_position.row,
                )
                .as_bytes(),
            );
        }
        return input;
    }

    let max_coordinate = if mouse.modes.utf8_mouse { 2_047 } else { 223 };
    let column = mouse_position.column.min(max_coordinate);
    let row = mouse_position.row.min(max_coordinate);
    let mut input = Vec::with_capacity(count * 6);
    for _ in 0..count {
        input.extend_from_slice(b"\x1b[M");
        push_mouse_coordinate(&mut input, 32 + button, mouse.modes.utf8_mouse);
        push_mouse_coordinate(&mut input, 32 + column as u16, mouse.modes.utf8_mouse);
        push_mouse_coordinate(&mut input, 32 + row as u16, mouse.modes.utf8_mouse);
    }
    input
}

fn push_mouse_coordinate(input: &mut Vec<u8>, value: u16, utf8: bool) {
    if utf8 {
        let mut buffer = [0; 4];
        let character = char::from_u32(u32::from(value)).expect("mouse coordinate is valid");
        input.extend_from_slice(character.encode_utf8(&mut buffer).as_bytes());
    } else {
        input.push(value as u8);
    }
}

fn mouse_button_code(button: MouseButton) -> Option<u16> {
    match button {
        MouseButton::Left => Some(0),
        MouseButton::Middle => Some(1),
        MouseButton::Right => Some(2),
        MouseButton::Navigate(_) => None,
    }
}

fn terminal_snap_to_device_pixel(value: f32, scale_factor: f32) -> f32 {
    let scale_factor = if scale_factor.is_finite() && scale_factor > 0.0 {
        scale_factor
    } else {
        1.0
    };
    let scaled = value * scale_factor;
    (scaled.abs() - 0.5).ceil().copysign(scaled) / scale_factor
}

fn terminal_grid_edge(origin: f32, advance: f32, index: usize, scale_factor: f32) -> f32 {
    terminal_snap_to_device_pixel(origin + advance * index as f32, scale_factor)
}

fn terminal_grid_index_at(
    coordinate: f32,
    origin: f32,
    advance: f32,
    scale_factor: f32,
    extent: f32,
) -> (usize, TerminalSelectionSide) {
    if coordinate < terminal_grid_edge(origin, advance, 0, scale_factor) {
        return (0, TerminalSelectionSide::Left);
    }

    let max_index = (extent.max(0.0) / advance.max(f32::EPSILON)).ceil() as usize + 2;
    let mut index = 0;
    while index + 1 < max_index
        && coordinate >= terminal_grid_edge(origin, advance, index + 1, scale_factor)
    {
        index += 1;
    }
    let left = terminal_grid_edge(origin, advance, index, scale_factor);
    let right = terminal_grid_edge(origin, advance, index + 1, scale_factor);
    let side = if coordinate >= (left + right) / 2.0 {
        TerminalSelectionSide::Right
    } else {
        TerminalSelectionSide::Left
    };
    (index, side)
}

fn terminal_columns_for_width(bounds: Bounds<gpui::Pixels>, metrics: TerminalMetrics) -> usize {
    let origin = f32::from(bounds.origin.x);
    let right = terminal_snap_to_device_pixel(f32::from(bounds.right()), metrics.scale_factor);
    let extent =
        (right - terminal_grid_edge(origin, metrics.cell_width, 0, metrics.scale_factor)).max(0.0);
    let estimate = (extent / metrics.cell_width.max(f32::EPSILON)).ceil() as usize + 2;
    (0..estimate)
        .take_while(|column| {
            terminal_grid_edge(
                origin,
                metrics.cell_width,
                *column + 1,
                metrics.scale_factor,
            ) <= right
        })
        .count()
}

fn terminal_resize_request_needed(
    requests: &mut BTreeMap<(ConnectionId, PaneId), TerminalResizeRequest>,
    connection_id: ConnectionId,
    pane_id: PaneId,
    terminal_id: TerminalId,
    current_size: TerminalSize,
    target: TerminalSize,
    window_active: bool,
) -> bool {
    let key = (connection_id, pane_id);
    if target == current_size {
        requests.remove(&key);
        return false;
    }
    if !window_active {
        return false;
    }

    let request = TerminalResizeRequest {
        terminal_id,
        target,
        observed_size: current_size,
    };
    if requests.get(&key) == Some(&request) {
        false
    } else {
        requests.insert(key, request);
        true
    }
}

fn terminal_lines_for_height(bounds: Bounds<gpui::Pixels>, metrics: TerminalMetrics) -> usize {
    let origin = f32::from(bounds.origin.y);
    let bottom = terminal_snap_to_device_pixel(f32::from(bounds.bottom()), metrics.scale_factor);
    let extent = (bottom
        - terminal_grid_edge(origin, metrics.line_height, 0, metrics.scale_factor))
    .max(0.0);
    let estimate = (extent / metrics.line_height.max(f32::EPSILON)).ceil() as usize + 2;
    (0..estimate)
        .take_while(|row| {
            terminal_grid_edge(origin, metrics.line_height, *row + 1, metrics.scale_factor)
                <= bottom
        })
        .count()
}

fn terminal_mouse_position(
    position: Point<gpui::Pixels>,
    bounds: Option<Bounds<gpui::Pixels>>,
    metrics: TerminalMetrics,
) -> TerminalMousePosition {
    let Some(bounds) = bounds else {
        return TerminalMousePosition {
            column: 1,
            row: 1,
            side: TerminalSelectionSide::Left,
        };
    };
    let position_x = f32::from(position.x);
    let position_y = f32::from(position.y);
    let origin_x = f32::from(bounds.origin.x);
    let origin_y = f32::from(bounds.origin.y);
    let (column, column_side) = terminal_grid_index_at(
        position_x,
        origin_x,
        metrics.cell_width,
        metrics.scale_factor,
        f32::from(bounds.size.width),
    );
    let (row, _) = terminal_grid_index_at(
        position_y,
        origin_y,
        metrics.line_height,
        metrics.scale_factor,
        f32::from(bounds.size.height),
    );
    let side = if position_x
        < terminal_grid_edge(origin_x, metrics.cell_width, 0, metrics.scale_factor)
        || position_y < terminal_grid_edge(origin_y, metrics.line_height, 0, metrics.scale_factor)
    {
        TerminalSelectionSide::Left
    } else if position_x
        >= terminal_snap_to_device_pixel(f32::from(bounds.right()), metrics.scale_factor)
        || position_y
            >= terminal_snap_to_device_pixel(f32::from(bounds.bottom()), metrics.scale_factor)
        || column_side == TerminalSelectionSide::Right
    {
        TerminalSelectionSide::Right
    } else {
        TerminalSelectionSide::Left
    };
    TerminalMousePosition {
        column: column + 1,
        row: row + 1,
        side,
    }
}

fn terminal_key_uses_text_input_handler(keystroke: &Keystroke) -> bool {
    // Special key names are always rendered through the key-down mapping.
    // Synthesized keystrokes (ui control / scenarios) carry a `key_char`
    // that repeats the key name; letting them take the text-input path
    // would type the literal word "return" into the shell instead of `\r`.
    if terminal_special_key_input_with_modes(
        &keystroke.key,
        keystroke.modifiers,
        crate::terminal::TerminalModes::default(),
    )
    .is_some()
    {
        return false;
    }
    (keystroke.key_char.as_deref().is_some_and(|character| {
        !character.is_empty() && character.chars().all(|character| !character.is_control())
    }) || keystroke.key.chars().count() == 1
        || keystroke.key == "space")
        && !keystroke.modifiers.control
        && !keystroke.modifiers.alt
        && !keystroke.modifiers.platform
        && !keystroke.modifiers.function
}

fn terminal_input_for_keystroke_with_modes(
    keystroke: &Keystroke,
    modes: TerminalModes,
) -> Option<String> {
    let modifiers = keystroke.modifiers;
    if modifiers.platform || modifiers.function {
        return None;
    }

    let key = keystroke.key.as_str();
    if let Some(input) = terminal_special_key_input_with_modes(key, modifiers, modes) {
        return Some(input);
    }
    if modifiers.control {
        return terminal_control_input(key, keystroke.key_char.as_deref());
    }

    let character = keystroke
        .key_char
        .as_deref()
        .filter(|character| !character.is_empty())
        .or_else(|| (key.chars().count() == 1).then_some(key))
        .or_else(|| (key == "space").then_some(" "))?;
    let mut input = String::new();
    if modifiers.alt {
        input.push('\u{1b}');
    }
    input.push_str(character);
    Some(input)
}

fn clipboard_text(item: gpui::ClipboardItem) -> Option<String> {
    item.entries.into_iter().find_map(|entry| match entry {
        gpui::ClipboardEntry::String(text) => Some(text.into_text()),
        gpui::ClipboardEntry::Image(_) | gpui::ClipboardEntry::ExternalPaths(_) => None,
    })
}

fn terminal_special_key_input_with_modes(
    key: &str,
    modifiers: gpui::Modifiers,
    modes: TerminalModes,
) -> Option<String> {
    let modifier = terminal_modifier_parameter(modifiers);
    match key {
        "enter" | "return" => Some("\r".to_owned()),
        "backspace" => Some(if modifiers.alt {
            "\u{1b}\u{7f}".to_owned()
        } else if modifiers.control {
            "\u{8}".to_owned()
        } else {
            "\u{7f}".to_owned()
        }),
        "tab" => {
            if modifiers.shift && !modifiers.control && !modifiers.alt {
                Some("\u{1b}[Z".to_owned())
            } else {
                Some("\t".to_owned())
            }
        }
        "escape" => Some("\u{1b}".to_owned()),
        "left" => Some(cursor_sequence('D', modifier, modes.application_cursor)),
        "right" => Some(cursor_sequence('C', modifier, modes.application_cursor)),
        "up" => Some(cursor_sequence('A', modifier, modes.application_cursor)),
        "down" => Some(cursor_sequence('B', modifier, modes.application_cursor)),
        "home" => Some(cursor_sequence('H', modifier, modes.application_cursor)),
        "end" => Some(cursor_sequence('F', modifier, modes.application_cursor)),
        "insert" => Some(numbered_sequence("2", '~', modifier)),
        "delete" => Some(numbered_sequence("3", '~', modifier)),
        "pageup" => Some(numbered_sequence("5", '~', modifier)),
        "pagedown" => Some(numbered_sequence("6", '~', modifier)),
        "f1" => Some(function_key_sequence(1, modifier)),
        "f2" => Some(function_key_sequence(2, modifier)),
        "f3" => Some(function_key_sequence(3, modifier)),
        "f4" => Some(function_key_sequence(4, modifier)),
        "f5" => Some(numbered_sequence("15", '~', modifier)),
        "f6" => Some(numbered_sequence("17", '~', modifier)),
        "f7" => Some(numbered_sequence("18", '~', modifier)),
        "f8" => Some(numbered_sequence("19", '~', modifier)),
        "f9" => Some(numbered_sequence("20", '~', modifier)),
        "f10" => Some(numbered_sequence("21", '~', modifier)),
        "f11" => Some(numbered_sequence("23", '~', modifier)),
        "f12" => Some(numbered_sequence("24", '~', modifier)),
        _ => None,
    }
}

fn terminal_control_input(key: &str, key_char: Option<&str>) -> Option<String> {
    let character = match key {
        "space" => ' ',
        _ => key_char
            .and_then(|value| value.chars().next())
            .or_else(|| key.chars().next())?,
    };
    let character = character.to_ascii_lowercase();
    let byte = match character {
        'a'..='z' => character as u8 & 0x1f,
        '@' | '2' | ' ' => 0,
        '[' | '3' => 0x1b,
        '\\' | '4' => 0x1c,
        ']' | '5' => 0x1d,
        '^' | '6' => 0x1e,
        '_' | '7' => 0x1f,
        '?' | '8' => 0x7f,
        _ => return None,
    };
    Some(char::from(byte).to_string())
}

fn terminal_modifier_parameter(modifiers: gpui::Modifiers) -> u8 {
    1 + u8::from(modifiers.shift) + 2 * u8::from(modifiers.alt) + 4 * u8::from(modifiers.control)
}

fn cursor_sequence(final_character: char, modifier: u8, application_cursor: bool) -> String {
    if modifier == 1 && application_cursor {
        format!("\u{1b}O{final_character}")
    } else if modifier == 1 {
        format!("\u{1b}[{final_character}")
    } else {
        format!("\u{1b}[1;{modifier}{final_character}")
    }
}

fn numbered_sequence(number: &str, final_character: char, modifier: u8) -> String {
    if modifier == 1 {
        format!("\u{1b}[{number}{final_character}")
    } else {
        format!("\u{1b}[{number};{modifier}{final_character}")
    }
}

fn function_key_sequence(number: u8, modifier: u8) -> String {
    if modifier == 1 {
        return match number {
            1 => "\u{1b}OP".to_owned(),
            2 => "\u{1b}OQ".to_owned(),
            3 => "\u{1b}OR".to_owned(),
            4 => "\u{1b}OS".to_owned(),
            _ => unreachable!("function key sequence only supports F1-F4"),
        };
    }
    let final_character = match number {
        1 => 'P',
        2 => 'Q',
        3 => 'R',
        4 => 'S',
        _ => unreachable!("function key sequence only supports F1-F4"),
    };
    format!("\u{1b}[1;{modifier}{final_character}")
}

#[cfg(test)]
fn is_terminal_cell_selected(
    snapshot: &TerminalSnapshot,
    selection: Option<TerminalSelection>,
    terminal_id: TerminalId,
    row: usize,
    column: usize,
) -> bool {
    let Some(selection) = selection else {
        return false;
    };
    if selection.terminal_id != terminal_id {
        return false;
    }
    let Some((start, end)) = selection_bounds(snapshot, selection) else {
        return false;
    };
    (start..=end).contains(&TerminalCellPosition {
        row: row as i32,
        column,
    })
}

fn selection_boundary_index(endpoint: TerminalSelectionEndpoint, columns: usize) -> i64 {
    i64::from(endpoint.position.row) * columns as i64
        + endpoint.position.column as i64
        + i64::from(endpoint.side == TerminalSelectionSide::Right)
}

fn terminal_selection_after_click(
    previous: Option<TerminalSelection>,
    terminal_id: TerminalId,
    endpoint: TerminalSelectionEndpoint,
    shift_held: bool,
) -> TerminalSelection {
    if shift_held
        && let Some(previous) = previous.filter(|selection| selection.terminal_id == terminal_id)
    {
        return TerminalSelection {
            terminal_id,
            anchor: previous.anchor,
            head: endpoint,
        };
    }
    TerminalSelection {
        terminal_id,
        anchor: endpoint,
        head: endpoint,
    }
}

fn selection_bounds(
    snapshot: &TerminalSnapshot,
    selection: TerminalSelection,
) -> Option<(TerminalCellPosition, TerminalCellPosition)> {
    let columns = snapshot.size.columns;
    let anchor_boundary = selection_boundary_index(selection.anchor, columns);
    let head_boundary = selection_boundary_index(selection.head, columns);
    let (start_endpoint, end_endpoint) = if anchor_boundary <= head_boundary {
        (selection.anchor, selection.head)
    } else {
        (selection.head, selection.anchor)
    };
    let first_row = -(snapshot.rows_before.len() as i64);
    let end_row = snapshot
        .size
        .lines
        .saturating_add(snapshot.rows_after.len()) as i64;
    let first_cell = first_row.saturating_mul(columns as i64);
    let end_cell = end_row.saturating_mul(columns as i64);
    let mut start = selection_boundary_index(start_endpoint, columns).clamp(first_cell, end_cell);
    let mut end = selection_boundary_index(end_endpoint, columns).clamp(first_cell, end_cell);
    if start >= end {
        return None;
    }

    while start < end {
        let Some(cell) = terminal_cell_at_linear_index(snapshot, start) else {
            break;
        };
        if cell.flags.leading_wide_spacer() {
            // This is the placeholder written at the end of a wrapped line
            // before a wide character starts on the next line. It is not the
            // second half of a character in this row.
            start = start.saturating_add(1);
        } else if cell.flags.wide_spacer() {
            // A trailing spacer belongs to the wide base immediately before
            // it. Normalize an endpoint landing on either half to the base.
            if start > first_cell
                && terminal_cell_at_linear_index(snapshot, start - 1)
                    .is_some_and(|cell| cell.flags.wide())
            {
                start -= 1;
            } else {
                start = start.saturating_add(1);
            }
        } else {
            break;
        }
    }
    if end > start
        && terminal_cell_at_linear_index(snapshot, end.saturating_sub(1))
            .is_some_and(|cell| cell.flags.leading_wide_spacer())
    {
        end = end.saturating_sub(1);
    }
    if end > start
        && terminal_cell_at_linear_index(snapshot, end.saturating_sub(1))
            .is_some_and(|cell| cell.flags.wide())
    {
        end = end.saturating_add(1).min(end_cell);
    }
    if start >= end {
        return None;
    }
    Some((
        TerminalCellPosition {
            row: start.div_euclid(columns as i64) as i32,
            column: start.rem_euclid(columns as i64) as usize,
        },
        TerminalCellPosition {
            row: (end - 1).div_euclid(columns as i64) as i32,
            column: (end - 1).rem_euclid(columns as i64) as usize,
        },
    ))
}

fn terminal_cell_at_linear_index(snapshot: &TerminalSnapshot, index: i64) -> Option<&TerminalCell> {
    let columns = snapshot.size.columns as i64;
    let row = i32::try_from(index.div_euclid(columns)).ok()?;
    let column = usize::try_from(index.rem_euclid(columns)).ok()?;
    snapshot.relative_row(row)?.get(column)
}

fn selected_terminal_text(snapshot: &TerminalSnapshot, selection: TerminalSelection) -> String {
    let Some((start, end)) = selection_bounds(snapshot, selection) else {
        return String::new();
    };
    let mut text = String::new();
    for row in start.row..=end.row {
        let Some(cells) = snapshot.relative_row(row) else {
            continue;
        };
        let first_column = if row == start.row { start.column } else { 0 };
        let last_column = if row == end.row {
            end.column
        } else {
            snapshot.size.columns.saturating_sub(1)
        };
        let line_start = text.len();
        for column in first_column..=last_column {
            let Some(cell) = cells.get(column) else {
                continue;
            };
            if cell.flags.wide_spacer() || cell.flags.leading_wide_spacer() {
                continue;
            }
            text.push(cell.character);
            text.extend(cell.zerowidth.iter().copied());
        }
        let line = &text[line_start..];
        let trimmed_len = line.trim_end_matches(' ').len();
        text.truncate(line_start + trimmed_len);
        if row != end.row {
            let wrapped = cells
                .get(snapshot.size.columns.saturating_sub(1))
                .is_some_and(|cell| cell.flags.wrapline());
            if !wrapped {
                text.push('\n');
            }
        }
    }
    text
}

#[allow(clippy::too_many_arguments)]
fn render_terminal_snapshot(
    snapshot: Arc<TerminalSnapshot>,
    selection: Option<TerminalSelection>,
    options: TerminalRenderOptions,
    font_family: &str,
    font_size: f32,
    ime_text: Option<String>,
    terminal_bounds: Arc<Mutex<BTreeMap<TerminalId, Bounds<gpui::Pixels>>>>,
    render_caches: TerminalRenderCaches,
    input_handler: Option<(Entity<WorkspaceView>, FocusHandle)>,
) -> AnyElement {
    TerminalRenderElement {
        snapshot,
        selection,
        options,
        font_family: font_family.to_owned(),
        font_size,
        ime_text,
        terminal_bounds,
        render_caches,
        input_handler,
    }
    .into_any_element()
}

#[allow(clippy::too_many_arguments)]
fn terminal_row_paint(
    snapshot: &TerminalSnapshot,
    row: i32,
    cells: &[TerminalCell],
    selected_bounds: Option<(TerminalCellPosition, TerminalCellPosition)>,
    options: TerminalRenderOptions,
    font_family: &str,
    font_size: f32,
    bounds: Bounds<gpui::Pixels>,
    window: &mut Window,
) -> TerminalRowPaint {
    let (chunks, backgrounds) =
        terminal_row_data_for_cells(snapshot, row, cells, selected_bounds, options, font_family);
    let mut text = Vec::new();
    for chunk in chunks {
        let target_width = f32::from(
            terminal_cell_bounds(
                bounds,
                options.metrics,
                0,
                chunk.start_column,
                chunk.span_columns,
            )
            .size
            .width,
        );
        let line = (!chunk.requires_cell_scaling).then(|| {
            shape_terminal_text_line(
                window,
                &chunk.text,
                &chunk.runs,
                font_size,
                options.metrics.cell_width * chunk.width_columns as f32,
                target_width,
            )
        });
        let should_shape_cells = chunk.requires_cell_scaling
            || line.as_ref().is_some_and(|line| {
                let natural_width = f32::from(line.width());
                natural_width.is_finite() && natural_width > target_width + 0.01
            });
        if should_shape_cells {
            let mut column = chunk.start_column;
            for cell in chunk.cells {
                let width = f32::from(
                    terminal_cell_bounds(bounds, options.metrics, 0, column, cell.width_columns)
                        .size
                        .width,
                );
                let line = shape_terminal_text_line(
                    window,
                    &cell.text,
                    std::slice::from_ref(&cell.run),
                    font_size,
                    width,
                    width,
                );
                text.push(TerminalTextPaint {
                    start_column: column,
                    line,
                });
                column += cell.width_columns;
            }
        } else if let Some(line) = line {
            text.push(TerminalTextPaint {
                start_column: chunk.start_column,
                line,
            });
        }
    }
    TerminalRowPaint {
        row,
        text,
        backgrounds,
    }
}

impl gpui::IntoElement for TerminalRenderElement {
    type Element = Self;

    fn into_element(self) -> Self::Element {
        self
    }
}

impl gpui::Element for TerminalRenderElement {
    type RequestLayoutState = ();
    type PrepaintState = TerminalPrepaintState;

    fn id(&self) -> Option<gpui::ElementId> {
        None
    }

    fn source_location(&self) -> Option<&'static std::panic::Location<'static>> {
        None
    }

    fn request_layout(
        &mut self,
        _id: Option<&gpui::GlobalElementId>,
        _inspector_id: Option<&gpui::InspectorElementId>,
        window: &mut Window,
        cx: &mut App,
    ) -> (gpui::LayoutId, Self::RequestLayoutState) {
        let mut style = gpui::Style::default();
        style.size.width = relative(1.).into();
        style.size.height =
            px(self.options.metrics.line_height * self.snapshot.size.lines as f32).into();
        (window.request_layout(style, [], cx), ())
    }

    fn prepaint(
        &mut self,
        _id: Option<&gpui::GlobalElementId>,
        _inspector_id: Option<&gpui::InspectorElementId>,
        bounds: Bounds<gpui::Pixels>,
        _request_layout: &mut Self::RequestLayoutState,
        window: &mut Window,
        _cx: &mut App,
    ) -> Self::PrepaintState {
        let prepaint_started = scroll_stats_enabled().then(Instant::now);
        self.terminal_bounds
            .lock()
            .expect("terminal bounds poisoned")
            .insert(self.snapshot.terminal_id, bounds);

        let selected_bounds = self
            .selection
            .filter(|selection| selection.terminal_id == self.snapshot.terminal_id)
            .and_then(|selection| selection_bounds(&self.snapshot, selection));
        let cache_key = TerminalRenderCacheKey {
            terminal_id: self.snapshot.terminal_id,
            snapshot_revision: self.snapshot.revision,
            viewport_position: self.snapshot.viewport_position,
            focused_cursor: (self.options.cursor_focused && self.snapshot.cursor.visible)
                .then(|| terminal_cursor_position(&self.snapshot)),
            font_family: self.font_family.clone(),
            font_size_bits: self.font_size.to_bits(),
            metrics: self.options.metrics,
            theme: self.options.theme,
            cursor_focused: self.options.cursor_focused,
            selection: self.selection,
            bounds_origin_x_bits: f32::from(bounds.origin.x).to_bits(),
            bounds_width_bits: f32::from(bounds.size.width).to_bits(),
        };
        let scroll_offset_rows = self.options.scroll_offset_rows;
        if scroll_offset_rows != 0.0 {
            scroll_stat_inc(&SCROLL_FRAMES);
        }
        let whole = scroll_offset_rows.trunc() as i32;
        let source_rows =
            terminal_visible_source_rows(self.snapshot.size.lines, scroll_offset_rows);
        let mut caches = self.render_caches.lock().expect("terminal cache poisoned");
        let cache = caches.entry(self.snapshot.terminal_id).or_default();
        if cache.key.as_ref() != Some(&cache_key) {
            let previous_key = cache.key.clone();
            let previous_viewport_position = previous_key
                .as_ref()
                .map(|key| key.viewport_position)
                .unwrap_or(self.snapshot.viewport_position);
            let rows_compatible = previous_key
                .as_ref()
                .is_some_and(|previous| previous.rows_compatible_with(&cache_key));
            let previous_rows = if rows_compatible {
                std::mem::take(&mut cache.rows)
            } else {
                cache.rows.clear();
                BTreeMap::new()
            };
            let mut previous_rows = previous_rows;
            let first = -(self.snapshot.rows_before.len() as i32);
            let end = self.snapshot.size.lines as i32 + self.snapshot.rows_after.len() as i32;
            for source_row in first..end {
                let Some(cells) = self.snapshot.relative_row_snapshot(source_row) else {
                    continue;
                };
                // A viewport ACK rebases row coordinates. Map the new row
                // back to the same physical grid row in the prior snapshot.
                // Socket/SSH deserialization necessarily creates new Arcs,
                // so equality is the cross-process structural-sharing key;
                // pointer equality keeps the in-process path essentially free.
                let Some(previous_source_row) = previous_cached_source_row(
                    source_row,
                    self.snapshot.viewport_position,
                    previous_viewport_position,
                ) else {
                    continue;
                };
                let Some(mut previous) =
                    previous_rows
                        .remove(&previous_source_row)
                        .filter(|previous| {
                            terminal_cached_row_cursor_compatible(
                                previous_key.as_ref().and_then(|key| key.focused_cursor),
                                cache_key.focused_cursor,
                                previous_source_row,
                                source_row,
                            ) && (Arc::ptr_eq(&previous.cells, cells)
                                || previous.cells.as_ref() == cells.as_ref())
                        })
                else {
                    continue;
                };
                previous.paint.row = source_row;
                cache.rows.insert(
                    source_row,
                    TerminalCachedRowPaint {
                        cells: cells.clone(),
                        paint: previous.paint,
                    },
                );
            }
            cache.key = Some(cache_key);
        }

        // Keep the immutable raw snapshot reserve wide, but shape only the
        // visible rows plus a small neighboring band. Subsequent scroll frames
        // reuse these prepared rows and shape only newly approached content.
        let snapshot_first = -(self.snapshot.rows_before.len() as i32);
        let snapshot_end = self.snapshot.size.lines as i32 + self.snapshot.rows_after.len() as i32;
        let prepared_rows =
            terminal_prepared_source_rows(source_rows.clone(), snapshot_first, snapshot_end);
        if std::env::var_os("WATER_DEBUG_SCROLL").is_some() {
            let missing: Vec<i32> = source_rows
                .clone()
                .filter(|r| self.snapshot.relative_row_snapshot(*r).is_none())
                .collect();
            // For each visible source row, report whether the paint can
            // resolve it and what text the first row contains.
            let first_row_text: String = source_rows
                .clone()
                .next()
                .and_then(|r| self.snapshot.relative_row_snapshot(r))
                .map(|cells| {
                    let s: String = cells.iter().map(|c| c.character).collect();
                    s.chars().take(40).collect()
                })
                .unwrap_or_default();
            tracing::warn!(
                target: "water::scroll",
                terminal_id = %self.snapshot.terminal_id,
                viewport = self.snapshot.viewport_position,
                history_len = self.snapshot.history_len,
                rows_before = self.snapshot.rows_before.len(),
                scroll_offset = scroll_offset_rows,
                source_rows = ?source_rows.clone(),
                prepared = ?prepared_rows.clone(),
                missing_source_rows = ?missing,
                first_visible_row_text = %first_row_text,
                "terminal paint: visible source rows vs available"
            );
        }
        if !prepared_rows.is_empty() {
            for source_row in prepared_rows {
                if cache.rows.contains_key(&source_row) {
                    continue;
                }
                let Some(cells) = self.snapshot.relative_row_snapshot(source_row) else {
                    continue;
                };
                let paint = terminal_row_paint(
                    &self.snapshot,
                    source_row,
                    cells,
                    selected_bounds,
                    self.options,
                    &self.font_family,
                    self.font_size,
                    bounds,
                    window,
                );
                cache.rows.insert(
                    source_row,
                    TerminalCachedRowPaint {
                        cells: cells.clone(),
                        paint,
                    },
                );
            }
        }

        let mut rows = Vec::with_capacity(source_rows.len());
        for source_row in source_rows {
            let Some(cached_row) = cache.rows.get(&source_row) else {
                continue;
            };
            let mut row = cached_row.paint.clone();
            row.row = source_row.saturating_add(whole);
            rows.push(row);
        }
        drop(caches);

        let ime_line = self.ime_text.as_deref().and_then(|text| {
            if text.is_empty() || !self.snapshot.cursor.visible {
                return None;
            }
            let (row, column) = terminal_cursor_position(&self.snapshot);
            let color = rgb(self.options.theme.terminal_foreground).into();
            let run = TextRun {
                len: text.len(),
                font: font(self.font_family.clone()),
                color,
                background_color: None,
                underline: Some(UnderlineStyle {
                    thickness: px(1.),
                    color: Some(color),
                    wavy: false,
                }),
                strikethrough: None,
            };
            scroll_stat_inc(&SCROLL_SHAPE_LINE_COUNT);
            let line = window.text_system().shape_line(
                SharedString::from(text.to_owned()),
                px(self.font_size),
                &[run],
                None,
            );
            Some((line, row, column))
        });

        if let Some(started) = prepaint_started {
            SCROLL_PREPAINT_MICROS
                .fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
        }
        TerminalPrepaintState { rows, ime_line }
    }

    fn paint(
        &mut self,
        _id: Option<&gpui::GlobalElementId>,
        _inspector_id: Option<&gpui::InspectorElementId>,
        bounds: Bounds<gpui::Pixels>,
        _request_layout: &mut Self::RequestLayoutState,
        prepaint: &mut Self::PrepaintState,
        window: &mut Window,
        cx: &mut App,
    ) {
        let TerminalRenderOptions {
            metrics,
            theme,
            cursor_focused,
            scroll_offset_rows,
        } = self.options;
        let whole = scroll_offset_rows.trunc() as i32;
        let fraction = scroll_offset_rows - whole as f32;
        window.paint_quad(fill(bounds, rgb(theme.terminal_background)));

        for row_paint in &prepaint.rows {
            for background in &row_paint.backgrounds {
                window.paint_quad(fill(
                    terminal_cell_bounds_for_row(
                        bounds,
                        metrics,
                        row_paint.row,
                        background.start_column,
                        background.width_columns,
                        fraction,
                    ),
                    rgb(background.color),
                ));
            }
            for text in &row_paint.text {
                let origin = terminal_cell_bounds_for_row(
                    bounds,
                    metrics,
                    row_paint.row,
                    text.start_column,
                    1,
                    fraction,
                )
                .origin;
                let _ = text.line.paint(
                    origin,
                    px(metrics.line_height),
                    TextAlign::Left,
                    None,
                    window,
                    cx,
                );
            }
        }

        if let Some((line, row, column)) = prepaint.ime_line.as_ref() {
            let origin = terminal_cell_bounds_for_row(
                bounds,
                metrics,
                (*row as i32).saturating_add(whole),
                *column,
                1,
                fraction,
            )
            .origin;
            let _ = line.paint(
                origin,
                px(metrics.line_height),
                TextAlign::Left,
                None,
                window,
                cx,
            );
        }

        if self.snapshot.cursor.visible && !cursor_focused {
            let (row, column) = terminal_cursor_position(&self.snapshot);
            let width_columns = self
                .snapshot
                .cell(row, column)
                .map(|cell| if cell.flags.wide() { 2 } else { 1 })
                .unwrap_or(1);
            // Draw the unfocused cursor as a hollow rectangle (outline only),
            // matching the behavior of most terminal emulators. The focused
            // cursor is already baked into the cell colors above, so this
            // block only applies to the non-focused pane.
            window.paint_quad(outline(
                terminal_cell_bounds_for_row(
                    bounds,
                    metrics,
                    (row as i32).saturating_add(whole),
                    column,
                    width_columns,
                    fraction,
                ),
                rgb(theme.inactive_cursor),
                gpui::BorderStyle::default(),
            ));
        }

        if let Some((view, focus_handle)) = self.input_handler.take() {
            window.handle_input(
                &focus_handle,
                TerminalInputHandler {
                    view,
                    terminal_id: self.snapshot.terminal_id,
                    element_bounds: bounds,
                    cursor: terminal_cursor_position(&self.snapshot),
                },
                cx,
            );
        }
    }
}

fn shape_terminal_text_line(
    window: &mut Window,
    text: &str,
    runs: &[TextRun],
    font_size: f32,
    force_width: f32,
    target_width: f32,
) -> ShapedLine {
    let text_system = window.text_system();
    let shape = |font_size: f32| {
        scroll_stat_inc(&SCROLL_SHAPE_LINE_COUNT);
        text_system.shape_line(
            SharedString::from(text.to_owned()),
            px(font_size),
            runs,
            Some(px(force_width)),
        )
    };
    let line = shape(font_size);
    let natural_width = f32::from(line.width());
    let scale = terminal_fit_scale(natural_width, target_width);
    if scale < 1.0 {
        shape((font_size * scale).max(0.5))
    } else {
        line
    }
}

fn terminal_fit_scale(natural_width: f32, target_width: f32) -> f32 {
    if target_width > 0.0 && natural_width.is_finite() && natural_width > target_width + 0.01 {
        (target_width / natural_width).clamp(0.05, 1.0)
    } else {
        1.0
    }
}

#[cfg(test)]
fn terminal_row_data(
    snapshot: &TerminalSnapshot,
    row: usize,
    selected_bounds: Option<(TerminalCellPosition, TerminalCellPosition)>,
    options: TerminalRenderOptions,
    font_family: &str,
) -> (Vec<TerminalTextChunk>, Vec<TerminalBackgroundSpan>) {
    let cells = snapshot.relative_row(row as i32).unwrap_or_default();
    terminal_row_data_for_cells(
        snapshot,
        row as i32,
        cells,
        selected_bounds,
        options,
        font_family,
    )
}

fn terminal_row_data_for_cells(
    snapshot: &TerminalSnapshot,
    row: i32,
    cells: &[TerminalCell],
    selected_bounds: Option<(TerminalCellPosition, TerminalCellPosition)>,
    options: TerminalRenderOptions,
    font_family: &str,
) -> (Vec<TerminalTextChunk>, Vec<TerminalBackgroundSpan>) {
    let fonts = {
        let normal = font(font_family.to_owned());
        [
            normal.clone(),
            normal.clone().italic(),
            normal.clone().bold(),
            normal.bold().italic(),
        ]
    };
    let mut chunks = Vec::new();
    let mut current_text = String::new();
    let mut current_runs = Vec::new();
    let mut current_cells = Vec::new();
    let mut current_requires_cell_scaling = false;
    let mut current_start = 0;

    let flush_chunk = |chunks: &mut Vec<TerminalTextChunk>,
                       current_text: &mut String,
                       current_runs: &mut Vec<TextRun>,
                       current_cells: &mut Vec<TerminalTextCell>,
                       current_requires_cell_scaling: &mut bool,
                       current_start: &mut usize| {
        if !current_text.is_empty() {
            let span_columns = current_cells.iter().map(|cell| cell.width_columns).sum();
            chunks.push(TerminalTextChunk {
                start_column: *current_start,
                width_columns: 1,
                span_columns,
                requires_cell_scaling: *current_requires_cell_scaling,
                text: std::mem::take(current_text),
                runs: std::mem::take(current_runs),
                cells: std::mem::take(current_cells),
            });
            *current_requires_cell_scaling = false;
        }
    };
    let mut backgrounds = Vec::new();

    for column in 0..snapshot.size.columns {
        let Some(cell) = cells.get(column) else {
            continue;
        };
        if cell.flags.wide_spacer() {
            flush_chunk(
                &mut chunks,
                &mut current_text,
                &mut current_runs,
                &mut current_cells,
                &mut current_requires_cell_scaling,
                &mut current_start,
            );
            continue;
        }

        let (mut foreground, mut background) = terminal_cell_colors(cell, options.theme);
        let selected = selected_bounds.is_some_and(|(start, end)| {
            (start..=end).contains(&TerminalCellPosition { row, column })
        });
        if selected {
            background = theme_color(options.theme.selection_background);
        }
        let cursor_at_cell = row >= 0
            && snapshot.cursor.visible
            && terminal_cursor_position(snapshot) == (row as usize, column);
        if cursor_at_cell && options.cursor_focused {
            foreground = theme_color(options.theme.cursor_foreground);
            background = theme_color(options.theme.cursor_background);
        }

        let width_columns = (if cell.flags.wide() { 2 } else { 1 })
            .min(snapshot.size.columns.saturating_sub(column));
        let background_color = color_to_rgb(background, false, options.theme);
        if background_color != options.theme.terminal_background {
            push_terminal_background(&mut backgrounds, column, width_columns, background_color);
        }
        if cell.flags.leading_wide_spacer() {
            flush_chunk(
                &mut chunks,
                &mut current_text,
                &mut current_runs,
                &mut current_cells,
                &mut current_requires_cell_scaling,
                &mut current_start,
            );
            continue;
        }

        let mut character = String::new();
        character.push(cell.character);
        character.extend(cell.zerowidth.iter().copied());
        let foreground_color = color_to_rgb(foreground, true, options.theme);
        let foreground_color = if cell.flags.dim() && !(cursor_at_cell && options.cursor_focused) {
            dim_terminal_color(foreground_color)
        } else {
            foreground_color
        };
        let color = rgb(foreground_color).into();
        let run = TextRun {
            len: character.len(),
            font: fonts[usize::from(cell.flags.italic()) + usize::from(cell.flags.bold()) * 2]
                .clone(),
            color,
            background_color: None,
            underline: cell.flags.underline().then(|| UnderlineStyle {
                thickness: px(1.),
                color: Some(color),
                wavy: false,
            }),
            strikethrough: cell.flags.strike().then(|| StrikethroughStyle {
                thickness: px(1.),
                color: Some(color),
            }),
        };
        let requires_cell_scaling = !character.is_ascii();
        let text_cell = TerminalTextCell {
            text: character.clone(),
            run: run.clone(),
            width_columns,
        };
        if cell.flags.wide() || requires_cell_scaling {
            flush_chunk(
                &mut chunks,
                &mut current_text,
                &mut current_runs,
                &mut current_cells,
                &mut current_requires_cell_scaling,
                &mut current_start,
            );
            chunks.push(TerminalTextChunk {
                start_column: column,
                width_columns,
                span_columns: width_columns,
                requires_cell_scaling: true,
                text: character,
                runs: vec![run],
                cells: vec![text_cell],
            });
        } else {
            if current_text.is_empty() {
                current_start = column;
            }
            current_text.push_str(&character);
            append_terminal_text_run(&mut current_runs, run);
            current_cells.push(text_cell);
        }
    }
    flush_chunk(
        &mut chunks,
        &mut current_text,
        &mut current_runs,
        &mut current_cells,
        &mut current_requires_cell_scaling,
        &mut current_start,
    );

    (chunks, backgrounds)
}

fn terminal_cell_colors(
    cell: &crate::terminal::TerminalCell,
    theme: ThemeColors,
) -> (TerminalColor, TerminalColor) {
    let mut foreground = cell.fg;
    let mut background = cell.bg;
    let default_colors = cell.fg == TerminalColor::Named { value: 256 }
        && cell.bg == TerminalColor::Named { value: 257 };
    if cell.flags.inverse() {
        std::mem::swap(&mut foreground, &mut background);
        if default_colors {
            foreground = theme_color(theme.inverse_foreground);
            background = theme_color(theme.inverse_background);
        }
    }
    (foreground, background)
}

fn append_terminal_text_run(runs: &mut Vec<TextRun>, run: TextRun) {
    if let Some(previous) = runs.last_mut()
        && previous.font == run.font
        && previous.color == run.color
        && previous.background_color == run.background_color
        && previous.underline == run.underline
        && previous.strikethrough == run.strikethrough
    {
        previous.len += run.len;
    } else {
        runs.push(run);
    }
}

fn push_terminal_background(
    backgrounds: &mut Vec<TerminalBackgroundSpan>,
    start_column: usize,
    width_columns: usize,
    color: u32,
) {
    if width_columns == 0 {
        return;
    }
    if let Some(previous) = backgrounds.last_mut()
        && previous.color == color
        && previous.start_column + previous.width_columns == start_column
    {
        previous.width_columns += width_columns;
    } else {
        backgrounds.push(TerminalBackgroundSpan {
            start_column,
            width_columns,
            color,
        });
    }
}

fn terminal_cell_bounds(
    bounds: Bounds<gpui::Pixels>,
    metrics: TerminalMetrics,
    row: usize,
    column: usize,
    width_columns: usize,
) -> Bounds<gpui::Pixels> {
    let left = terminal_grid_edge(
        f32::from(bounds.origin.x),
        metrics.cell_width,
        column,
        metrics.scale_factor,
    );
    let right = terminal_grid_edge(
        f32::from(bounds.origin.x),
        metrics.cell_width,
        column.saturating_add(width_columns),
        metrics.scale_factor,
    );
    let top = terminal_grid_edge(
        f32::from(bounds.origin.y),
        metrics.line_height,
        row,
        metrics.scale_factor,
    );
    let bottom = terminal_grid_edge(
        f32::from(bounds.origin.y),
        metrics.line_height,
        row.saturating_add(1),
        metrics.scale_factor,
    );
    Bounds::new(
        point(px(left), px(top)),
        size(px((right - left).max(0.0)), px((bottom - top).max(0.0))),
    )
}

fn terminal_row_position(row: i32, fractional_scroll_offset: f32) -> f32 {
    row as f32 + fractional_scroll_offset
}

fn terminal_visible_source_rows(
    screen_lines: usize,
    scroll_offset_rows: f32,
) -> std::ops::Range<i32> {
    let whole = scroll_offset_rows.trunc() as i32;
    let fraction = scroll_offset_rows - whole as f32;
    let first = -whole - i32::from(fraction > 0.0);
    let end = screen_lines as i32 - whole + i32::from(fraction < 0.0);
    first..end
}

/// Maps a pointer Y coordinate back to the exact source row painted beneath
/// it. Rendering shifts the snapped row grid by the fractional part of the
/// smooth-scroll offset, so hit-testing must undo that shift before choosing
/// a grid row. Otherwise the lower half of a visually shifted row selects its
/// previous neighbor.
fn terminal_source_row_at(
    position_y: gpui::Pixels,
    bounds: Option<Bounds<gpui::Pixels>>,
    metrics: TerminalMetrics,
    screen_lines: usize,
    scroll_offset_rows: f32,
) -> i32 {
    let visible = terminal_visible_source_rows(screen_lines, scroll_offset_rows);
    if visible.is_empty() {
        return 0;
    }
    let Some(bounds) = bounds else {
        return visible.start;
    };

    let whole = scroll_offset_rows.trunc() as i32;
    let fraction = scroll_offset_rows - whole as f32;
    let origin_y = f32::from(bounds.origin.y);
    let translated_y = f32::from(position_y) - fraction * metrics.line_height;
    let first_edge = terminal_grid_edge(origin_y, metrics.line_height, 0, metrics.scale_factor);
    let painted_row = if translated_y < first_edge {
        -1
    } else {
        let (row, _) = terminal_grid_index_at(
            translated_y,
            origin_y,
            metrics.line_height,
            metrics.scale_factor,
            f32::from(bounds.size.height),
        );
        i32::try_from(row).unwrap_or(i32::MAX)
    };
    painted_row
        .saturating_sub(whole)
        .max(visible.start)
        .min(visible.end.saturating_sub(1))
}

fn terminal_prepared_source_rows(
    visible_rows: std::ops::Range<i32>,
    snapshot_first: i32,
    snapshot_end: i32,
) -> std::ops::Range<i32> {
    if visible_rows.is_empty() {
        return 0..0;
    }
    visible_rows
        .start
        .saturating_sub(PREPARED_ROW_LOOKAHEAD)
        .max(snapshot_first)
        ..visible_rows
            .end
            .saturating_add(PREPARED_ROW_LOOKAHEAD)
            .min(snapshot_end)
}

/// Maps a row in a replacement snapshot to the same physical grid row in
/// the previous snapshot. A positive viewport ACK shifts visible content
/// toward the prior snapshot's negative overscan rows.
fn previous_cached_source_row(
    source_row: i32,
    viewport_position: i64,
    previous_viewport_position: i64,
) -> Option<i32> {
    i64::from(source_row)
        .saturating_sub(viewport_position)
        .saturating_add(previous_viewport_position)
        .try_into()
        .ok()
}

fn terminal_cached_row_cursor_compatible(
    previous_cursor: Option<(usize, usize)>,
    current_cursor: Option<(usize, usize)>,
    previous_source_row: i32,
    current_source_row: i32,
) -> bool {
    let cursor_is_on_row = |cursor: Option<(usize, usize)>, row: i32| {
        cursor.and_then(|(cursor_row, _)| i32::try_from(cursor_row).ok()) == Some(row)
    };
    !cursor_is_on_row(previous_cursor, previous_source_row)
        && !cursor_is_on_row(current_cursor, current_source_row)
}

/// Positions a visible or overscan row while preserving the device-snapped
/// geometry of the normal grid and adding only the fractional wheel movement.
fn terminal_cell_bounds_for_row(
    bounds: Bounds<gpui::Pixels>,
    metrics: TerminalMetrics,
    row: i32,
    column: usize,
    width_columns: usize,
    fractional_scroll_offset: f32,
) -> Bounds<gpui::Pixels> {
    let row_bounds = if row >= 0 {
        terminal_cell_bounds(bounds, metrics, row as usize, column, width_columns)
    } else {
        let first = terminal_cell_bounds(bounds, metrics, 0, column, width_columns);
        Bounds::new(
            point(
                first.origin.x,
                px(f32::from(first.origin.y) + row as f32 * metrics.line_height),
            ),
            first.size,
        )
    };
    let fractional_offset = terminal_row_position(row, fractional_scroll_offset) - row as f32;
    Bounds::new(
        point(
            row_bounds.origin.x,
            px(f32::from(row_bounds.origin.y) + fractional_offset * metrics.line_height),
        ),
        row_bounds.size,
    )
}

fn terminal_cursor_position(snapshot: &TerminalSnapshot) -> (usize, usize) {
    let row = snapshot
        .cursor
        .row
        .min(snapshot.size.lines.saturating_sub(1));
    let mut column = snapshot
        .cursor
        .column
        .min(snapshot.size.columns.saturating_sub(1));
    if snapshot
        .cell(row, column)
        .is_some_and(|cell| cell.flags.wide_spacer())
    {
        column = column.saturating_sub(1);
    }
    (row, column)
}

fn color_to_rgb(color: TerminalColor, foreground: bool, theme: ThemeColors) -> u32 {
    match color {
        TerminalColor::Rgb { red, green, blue } => {
            (u32::from(red) << 16) | (u32::from(green) << 8) | u32::from(blue)
        }
        TerminalColor::Named { value } => match value {
            0..=15 => basic_color(value as usize),
            256 if foreground => theme.terminal_foreground,
            257 if !foreground => theme.terminal_background,
            258 => theme.terminal_foreground,
            259 => theme.terminal_background,
            _ => theme.ui_foreground,
        },
        TerminalColor::Indexed { value } => indexed_color(value),
    }
}

fn theme_color(value: u32) -> TerminalColor {
    TerminalColor::Rgb {
        red: ((value >> 16) & 0xff) as u8,
        green: ((value >> 8) & 0xff) as u8,
        blue: (value & 0xff) as u8,
    }
}

fn dim_terminal_color(color: u32) -> u32 {
    const DIM_FACTOR: f32 = 0.66;
    let channel = |shift: u32| (((color >> shift) & 0xff) as f32 * DIM_FACTOR).round() as u32;
    (channel(16) << 16) | (channel(8) << 8) | channel(0)
}

fn basic_color(index: usize) -> u32 {
    // Match the 16-color palette from kitty_normal.conf so ANSI output and
    // the Water chrome share the same visual language.
    const COLORS: [u32; 16] = [
        0x000000, 0xaa0000, 0x00aa00, 0xf57900, 0x1e8acb, 0xaa00aa, 0x00aaaa, 0xaaaaaa, 0x555555,
        0xff5555, 0x339966, 0xffff55, 0x729fcf, 0xd530c6, 0x55ffff, 0xffffff,
    ];
    COLORS[index.min(COLORS.len() - 1)]
}

fn indexed_color(index: u8) -> u32 {
    if index < 16 {
        return basic_color(index as usize);
    }
    if (16..=231).contains(&index) {
        let index = index - 16;
        let red = index / 36;
        let green = (index % 36) / 6;
        let blue = index % 6;
        let channel = |value: u8| {
            if value == 0 {
                0
            } else {
                55 + 40 * u32::from(value)
            }
        };
        return (channel(red) << 16) | (channel(green) << 8) | channel(blue);
    }
    let gray = 8 + 10 * u32::from(index - 232);
    (gray << 16) | (gray << 8) | gray
}

impl Render for WorkspaceView {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        #[cfg(target_os = "macos")]
        if !window.is_fullscreen() {
            // AppKit keeps its standard traffic lights even with a transparent
            // titlebar. Park them off-canvas so the custom controls below are
            // the only visible controls in the integrated titlebar.
            window.set_traffic_light_position(point(px(-100.), px(9.)));
        }

        let metrics = self.measured_terminal_metrics(window);
        self.terminal_metrics = metrics;
        self.input_handler_terminal = self
            .active_terminal_snapshot()
            .map(|snapshot| snapshot.terminal_id);
        let theme = self.config.theme.colors();
        self.split_bounds
            .lock()
            .expect("split bounds poisoned")
            .clear();

        let window_active = window.is_window_active() && self.focus_handle.is_focused(window);
        let content = div()
            .flex_1()
            .flex()
            .flex_col()
            .min_w(px(0.))
            .min_h(px(0.))
            .overflow_hidden()
            .bg(rgb(theme.terminal_background))
            .child(self.render_active_tab(window_active, metrics, theme, cx));

        let action_view = cx.entity();
        let mut main_content = div()
            .flex_1()
            .min_w(px(0.))
            .min_h(px(0.))
            .flex()
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(|this, event: &MouseDownEvent, window, cx| {
                    if this.has_transient_ui() {
                        cx.stop_propagation();
                        return;
                    }
                    this.focus_handle.focus(window, cx);
                    this.begin_window_drag(event.position);
                    cx.stop_propagation();
                }),
            );
        if !self.sidebar_collapsed {
            main_content = main_content.child(self.render_sidebar(theme, cx));
        }
        let main_content = main_content.child(content);
        let overlay = self
            .render_dialog(theme, cx)
            .or_else(|| self.render_context_menu(theme, cx));
        let mut root = div()
            .size_full()
            .flex()
            .flex_col()
            .on_action(|_: &HideWindow, window, _cx| {
                window.remove_window();
            })
            .on_action(|_: &MinimizeWindow, window, _cx| {
                window.minimize_window();
            })
            .on_action(|_: &IgnoreQuit, _window, _cx| {})
            .on_action({
                let view = action_view.clone();
                move |_: &NewTerminalTab, _window, cx| {
                    view.update(cx, |workspace, cx| {
                        if !workspace.has_transient_ui() {
                            workspace.new_terminal_tab(cx);
                        }
                    });
                }
            })
            .on_action({
                let view = action_view.clone();
                move |_: &SplitRight, _window, cx| {
                    view.update(cx, |workspace, cx| {
                        if !workspace.has_transient_ui() {
                            workspace.split_active_pane(SplitDirection::Right, cx);
                        }
                    });
                }
            })
            .on_action({
                let view = action_view.clone();
                move |_: &SplitDown, _window, cx| {
                    view.update(cx, |workspace, cx| {
                        if !workspace.has_transient_ui() {
                            workspace.split_active_pane(SplitDirection::Down, cx);
                        }
                    });
                }
            })
            .on_action({
                let view = action_view.clone();
                move |_: &NewWorkspace, _window, cx| {
                    view.update(cx, |workspace, cx| {
                        if !workspace.has_transient_ui() {
                            workspace.dispatch_new_workspace(cx);
                        }
                    });
                }
            })
            .on_action({
                let view = action_view.clone();
                move |_: &ToggleSidebar, _window, cx| {
                    view.update(cx, |workspace, cx| {
                        if !workspace.has_transient_ui() {
                            workspace.toggle_sidebar(cx);
                        }
                    });
                }
            })
            .on_action({
                let view = action_view.clone();
                move |_: &ActivateTab1, _window, cx| {
                    view.update(cx, |workspace, cx| {
                        if !workspace.has_transient_ui() {
                            workspace.activate_tab_index(0, cx);
                        }
                    });
                }
            })
            .on_action({
                let view = action_view.clone();
                move |_: &ActivateTab2, _window, cx| {
                    view.update(cx, |workspace, cx| {
                        if !workspace.has_transient_ui() {
                            workspace.activate_tab_index(1, cx);
                        }
                    });
                }
            })
            .on_action({
                let view = action_view.clone();
                move |_: &ActivateTab3, _window, cx| {
                    view.update(cx, |workspace, cx| {
                        if !workspace.has_transient_ui() {
                            workspace.activate_tab_index(2, cx);
                        }
                    });
                }
            })
            .on_action({
                let view = action_view.clone();
                move |_: &ActivateTab4, _window, cx| {
                    view.update(cx, |workspace, cx| {
                        if !workspace.has_transient_ui() {
                            workspace.activate_tab_index(3, cx);
                        }
                    });
                }
            })
            .on_action({
                let view = action_view.clone();
                move |_: &ActivateTab5, _window, cx| {
                    view.update(cx, |workspace, cx| {
                        if !workspace.has_transient_ui() {
                            workspace.activate_tab_index(4, cx);
                        }
                    });
                }
            })
            .on_action({
                let view = action_view.clone();
                move |_: &ActivateTab6, _window, cx| {
                    view.update(cx, |workspace, cx| {
                        if !workspace.has_transient_ui() {
                            workspace.activate_tab_index(5, cx);
                        }
                    });
                }
            })
            .on_action({
                let view = action_view.clone();
                move |_: &ActivateTab7, _window, cx| {
                    view.update(cx, |workspace, cx| {
                        if !workspace.has_transient_ui() {
                            workspace.activate_tab_index(6, cx);
                        }
                    });
                }
            })
            .on_action({
                let view = action_view.clone();
                move |_: &ActivateTab8, _window, cx| {
                    view.update(cx, |workspace, cx| {
                        if !workspace.has_transient_ui() {
                            workspace.activate_tab_index(7, cx);
                        }
                    });
                }
            })
            .on_action({
                let view = action_view.clone();
                move |_: &ActivateTab9, _window, cx| {
                    view.update(cx, |workspace, cx| {
                        if !workspace.has_transient_ui() {
                            workspace.activate_tab_index(8, cx);
                        }
                    });
                }
            })
            .on_action({
                let view = action_view.clone();
                move |_: &ActivateTab10, _window, cx| {
                    view.update(cx, |workspace, cx| {
                        if !workspace.has_transient_ui() {
                            workspace.activate_tab_index(9, cx);
                        }
                    });
                }
            })
            .on_action({
                let view = action_view.clone();
                move |_: &NextTab, _window, cx| {
                    view.update(cx, |workspace, cx| {
                        if !workspace.has_transient_ui() {
                            workspace.activate_relative_tab(1, cx);
                        }
                    });
                }
            })
            .on_action({
                let view = action_view.clone();
                move |_: &PreviousTab, _window, cx| {
                    view.update(cx, |workspace, cx| {
                        if !workspace.has_transient_ui() {
                            workspace.activate_relative_tab(-1, cx);
                        }
                    });
                }
            })
            .on_action({
                let view = action_view.clone();
                move |_: &NextWorkspace, _window, cx| {
                    view.update(cx, |workspace, cx| {
                        if !workspace.has_transient_ui() {
                            workspace.activate_relative_workspace(1, cx);
                        }
                    });
                }
            })
            .on_action({
                let view = action_view.clone();
                move |_: &PreviousWorkspace, _window, cx| {
                    view.update(cx, |workspace, cx| {
                        if !workspace.has_transient_ui() {
                            workspace.activate_relative_workspace(-1, cx);
                        }
                    });
                }
            })
            .on_action(cx.listener(|workspace, _: &RenameWorkspace, window, cx| {
                workspace.begin_rename_active_workspace(window, cx);
            }))
            .on_action(cx.listener(|workspace, _: &RenameTab, window, cx| {
                workspace.begin_rename_active_tab(window, cx);
            }))
            .track_focus(&self.focus_handle)
            .on_key_down(cx.listener(|this, event: &KeyDownEvent, _window, cx| {
                this.handle_key_down(event, cx);
            }))
            .bg(rgb(theme.terminal_background))
            .text_size(px(self.config.ui.font_size))
            .text_color(rgb(theme.ui_foreground))
            .child(workspace_mouse_event_observer(cx.entity()))
            .child(self.render_titlebar(theme, cx))
            .child(main_content);
        if let Some(overlay) = overlay {
            root = root.child(overlay);
        }
        root
    }
}

#[allow(dead_code)]
fn _pane_id_is_explicitly_typed(_pane_id: PaneId) {}

#[allow(dead_code)]
fn _tab_dump_is_a_projection(_tab: &TabDump) {}

/// Which tab-strip edges have hidden content beyond the viewport. `max_x`
/// is the scrollable overflow magnitude (>= 0) and `offset_x` the current
/// scroll position (in `[-max_x, 0]`) recorded by the last layout pass.
fn tab_bar_edge_indicators(max_x: f32, offset_x: f32) -> (bool, bool) {
    let overflow = max_x > 1.0;
    (
        overflow && offset_x < -1.0,
        overflow && offset_x > -max_x + 1.0,
    )
}

/// Blend two packed RGB colors; `factor` weights `from` (1.0 keeps `from`).
fn mix_rgb(from: u32, to: u32, factor: f32) -> u32 {
    let channel = |value: u32, shift: u32| ((value >> shift) & 0xff) as f32;
    let blended = |shift: u32| {
        let mixed = channel(from, shift) * factor + channel(to, shift) * (1.0 - factor);
        mixed.round().clamp(0.0, 255.0) as u32
    };
    (blended(16) << 16) | (blended(8) << 8) | blended(0)
}

/// Maximum per-channel distance between two packed RGB colors (0..=255).
fn channel_distance(a: u32, b: u32) -> u32 {
    let channel = |value: u32, shift: u32| ((value >> shift) & 0xff) as i32;
    (0..3)
        .map(|shift| (channel(a, shift * 8) - channel(b, shift * 8)).abs())
        .max()
        .unwrap_or(0) as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpui::Modifiers;

    fn keystroke(key: &str, key_char: Option<&str>, modifiers: Modifiers) -> Keystroke {
        Keystroke {
            key: key.to_owned(),
            key_char: key_char.map(str::to_owned),
            modifiers,
        }
    }

    fn endpoint(row: i32, column: usize, side: TerminalSelectionSide) -> TerminalSelectionEndpoint {
        TerminalSelectionEndpoint {
            position: TerminalCellPosition { row, column },
            side,
        }
    }

    fn workspace_dump(id: WorkspaceId) -> WorkspaceDump {
        WorkspaceDump {
            id,
            title: format!("Workspace {id}"),
            active_tab: None,
            tabs: Vec::new(),
        }
    }

    fn model_snapshot(
        workspaces: Vec<WorkspaceDump>,
        active_workspace: Option<WorkspaceId>,
    ) -> ModelSnapshot {
        ModelSnapshot {
            state_revision: 1,
            workspace: workspaces.first().cloned(),
            workspaces,
            active_workspace,
            focused_pane: None,
            agents: Vec::new(),
        }
    }

    #[test]
    fn workspace_selection_preserves_local_choice_and_falls_back_safely() {
        let first = workspace_dump(WorkspaceId::new(1));
        let second = workspace_dump(WorkspaceId::new(2));
        let snapshot = model_snapshot(vec![first.clone(), second.clone()], Some(second.id));

        assert_eq!(
            workspace_selection_after_snapshot(Some(first.id), &snapshot),
            Some(first.id)
        );
        assert_eq!(
            workspace_selection_after_snapshot(Some(WorkspaceId::new(99)), &snapshot),
            Some(second.id)
        );

        let no_active = model_snapshot(vec![first.clone(), second], None);
        assert_eq!(
            workspace_selection_after_snapshot(Some(WorkspaceId::new(99)), &no_active),
            Some(first.id)
        );
        let empty = model_snapshot(Vec::new(), None);
        assert_eq!(
            workspace_selection_after_snapshot(Some(first.id), &empty),
            None
        );
    }

    #[test]
    fn pane_resize_requests_are_connection_scoped_and_reassert_after_supersession() {
        let pane_id = PaneId::new(1);
        let terminal_id = TerminalId::new(1);
        let local = ConnectionId::new(1);
        let remote = ConnectionId::new(2);
        let initial = TerminalSize::new(80, 24);
        let first_window = TerminalSize::new(100, 30);
        let second_window = TerminalSize::new(120, 40);
        let mut requests = BTreeMap::new();

        assert!(terminal_resize_request_needed(
            &mut requests,
            local,
            pane_id,
            terminal_id,
            initial,
            first_window,
            true,
        ));
        assert!(!terminal_resize_request_needed(
            &mut requests,
            local,
            pane_id,
            terminal_id,
            initial,
            first_window,
            true,
        ));
        assert!(
            terminal_resize_request_needed(
                &mut requests,
                local,
                pane_id,
                TerminalId::new(2),
                initial,
                first_window,
                true,
            ),
            "replacing the terminal surface in a pane must create a fresh request"
        );
        assert!(
            terminal_resize_request_needed(
                &mut requests,
                remote,
                pane_id,
                terminal_id,
                initial,
                first_window,
                true,
            ),
            "colliding remote pane/terminal IDs must not inherit the local request"
        );

        assert!(
            terminal_resize_request_needed(
                &mut requests,
                local,
                pane_id,
                TerminalId::new(2),
                second_window,
                first_window,
                true,
            ),
            "the active window must reassert its size after another window resized the PTY"
        );
        assert!(!terminal_resize_request_needed(
            &mut requests,
            local,
            pane_id,
            terminal_id,
            first_window,
            first_window,
            false,
        ));
        assert!(!requests.contains_key(&(local, pane_id)));
    }

    #[test]
    fn fractional_scroll_repaints_without_requesting_a_whole_row() {
        let mut state = TerminalScrollState::new(10);
        assert_eq!(state.accumulate(0.75), (None, true));
        assert!((state.visual_unacked_rows - 0.75).abs() < f32::EPSILON);
        assert_eq!(state.requested_viewport_position, 10);
    }

    #[test]
    fn fractional_scroll_continues_while_a_worker_request_is_pending() {
        let mut state = TerminalScrollState::new(10);
        assert_eq!(state.accumulate(0.75), (None, true));
        assert_eq!(state.accumulate(0.5), (Some(11), true));
        assert!((state.visual_unacked_rows - 1.25).abs() < f32::EPSILON);
        assert_eq!(state.requested_viewport_position, 11);
    }

    #[test]
    fn snapshot_ack_rebases_without_moving_the_visual_position() {
        let mut state = TerminalScrollState::new(10);
        assert_eq!(state.accumulate(1.25), (Some(11), true));
        let before = state.observed_viewport_position as f32 + state.visual_unacked_rows;

        let mut snapshot = TerminalSnapshot::empty(TerminalId::new(1), TerminalSize::new(8, 4));
        snapshot.viewport_position = 11;
        snapshot.rows_before.push(Arc::from(
            vec![TerminalCell::default(); 8].into_boxed_slice(),
        ));
        snapshot.rows_after.push(Arc::from(
            vec![TerminalCell::default(); 8].into_boxed_slice(),
        ));
        reconcile_visual_scroll(&mut state, &snapshot);

        let after = state.observed_viewport_position as f32 + state.visual_unacked_rows;
        assert!((state.visual_unacked_rows - 0.25).abs() < f32::EPSILON);
        assert!((before - after).abs() < f32::EPSILON);
    }

    #[test]
    fn local_viewport_reconciliation_allows_scrolling_past_first_batch() {
        let terminal_id = TerminalId::new(1);
        let mut states = BTreeMap::new();
        let mut state = TerminalScrollState::new(0);
        assert_eq!(state.accumulate(100.0), (Some(100), true));
        states.insert(terminal_id, state);

        let previous = TerminalSnapshot::empty(terminal_id, TerminalSize::new(8, 4));
        let mut next = previous.clone();
        next.viewport_position = 100;
        let mut selection = None;
        reconcile_local_viewport_snapshot(
            &mut selection,
            &mut states,
            terminal_id,
            &previous,
            &next,
        );

        let state = states.get_mut(&terminal_id).unwrap();
        assert_eq!(state.observed_viewport_position, 100);
        assert_eq!(state.visual_unacked_rows, 0.0);
        assert_eq!(state.accumulate(100.0), (Some(200), true));
    }

    #[test]
    fn reversing_before_ack_corrects_the_absolute_visual_target() {
        let mut state = TerminalScrollState::new(10);
        assert_eq!(state.accumulate(1.4), (Some(11), true));
        assert_eq!(state.accumulate(-0.9), (Some(10), true));
        assert!((state.visual_unacked_rows - 0.5).abs() < f32::EPSILON);
        assert_eq!(state.requested_viewport_position, 10);
    }

    #[test]
    fn live_bottom_discards_hidden_scroll_debt_before_reversing() {
        let mut state = TerminalScrollState::new(10);
        assert_eq!(state.accumulate(-20.0), (Some(-10), true));

        // The snapshot says the live bottom is already visible. The first
        // opposite delta must move immediately instead of paying back -20.
        assert_eq!(
            state.accumulate_with_boundaries(0.5, false, true),
            (None, true)
        );
        assert!((state.visual_unacked_rows - 0.5).abs() < f32::EPSILON);
        assert_eq!(state.requested_viewport_position, 10);
    }

    #[test]
    fn history_start_discards_hidden_scroll_debt_before_reversing() {
        let mut state = TerminalScrollState::new(10);
        assert_eq!(state.accumulate(20.0), (Some(30), true));

        // The same rule is symmetric at the oldest history row.
        assert_eq!(
            state.accumulate_with_boundaries(-0.5, true, false),
            (None, true)
        );
        assert!((state.visual_unacked_rows + 0.5).abs() < f32::EPSILON);
        assert_eq!(state.requested_viewport_position, 10);
    }

    #[test]
    fn deltas_further_into_a_known_boundary_are_not_accumulated() {
        let mut bottom = TerminalScrollState::new(0);
        assert_eq!(
            bottom.accumulate_with_boundaries(-100.0, false, true),
            (None, false)
        );
        assert_eq!(bottom.visual_unacked_rows, 0.0);

        let mut top = TerminalScrollState::new(100);
        assert_eq!(
            top.accumulate_with_boundaries(100.0, true, false),
            (None, false)
        );
        assert_eq!(top.visual_unacked_rows, 0.0);
    }

    #[test]
    fn touch_phase_distinguishes_trackpad_gestures_from_mouse_pixels() {
        let started = ScrollWheelEvent {
            delta: ScrollDelta::Pixels(point(px(0.0), px(1.0))),
            touch_phase: TouchPhase::Started,
            ..ScrollWheelEvent::default()
        };
        let moved = ScrollWheelEvent {
            touch_phase: TouchPhase::Moved,
            ..started.clone()
        };
        let lines = ScrollWheelEvent {
            delta: ScrollDelta::Lines(point(0.0, 1.0)),
            touch_phase: TouchPhase::Started,
            ..ScrollWheelEvent::default()
        };

        assert_eq!(
            terminal_scroll_input_kind(&started, false),
            TerminalScrollInputKind::TrackpadGesture
        );
        assert_eq!(
            terminal_scroll_input_kind(&moved, true),
            TerminalScrollInputKind::TrackpadGesture
        );
        assert_eq!(
            terminal_scroll_input_kind(&moved, false),
            TerminalScrollInputKind::MouseWheel
        );
        assert_eq!(
            terminal_scroll_input_kind(&lines, true),
            TerminalScrollInputKind::MouseWheel
        );
    }

    #[test]
    fn mouse_scroll_animation_uses_elapsed_time_and_finishes_exactly() {
        let started_at = Instant::now();
        let animation = TerminalMouseScrollAnimation {
            pane_id: PaneId::new(1),
            start_position: 10.6,
            target_position: 13.0,
            started_at,
        };

        assert_eq!(animation.position_at(started_at), (10.6, false));
        let (halfway, halfway_done) =
            animation.position_at(started_at + MOUSE_SCROLL_ANIMATION_DURATION / 2);
        assert!(!halfway_done);
        assert!(halfway > 10.6 && halfway < 13.0);

        // Skipping intermediate callbacks does not slow the animation: its
        // final position is derived from elapsed time, not a frame counter.
        let (finished, done) = animation.position_at(started_at + MOUSE_SCROLL_ANIMATION_DURATION);
        assert!(done);
        assert!((finished - 13.0).abs() < f32::EPSILON);
    }

    #[test]
    fn ui_viewport_requests_are_latest_wins_within_a_frame() {
        let terminal_id = TerminalId::new(1);
        let pane_id = PaneId::new(2);
        let mut pending = BTreeMap::new();
        for target in 1..=20 {
            record_latest_viewport_request(&mut pending, terminal_id, pane_id, target);
        }
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[&terminal_id], (pane_id, 20));
    }

    #[test]
    fn delayed_worker_ack_never_freezes_or_moves_the_visual_viewport_backwards() {
        let mut state = TerminalScrollState::new(10);
        let mut last_visual_position = 10.0;

        // All gesture events arrive before the simulated worker ACK at 40 ms.
        for (_at_ms, delta) in [
            (0_u64, 0.21_f32),
            (5, 0.32),
            (11, 0.41),
            (19, 0.37),
            (28, 0.48),
        ] {
            let (_, repaint) = state.accumulate(delta);
            let visual_position =
                state.observed_viewport_position as f32 + state.visual_unacked_rows;
            assert!(repaint);
            assert!(visual_position > last_visual_position);
            last_visual_position = visual_position;
        }
        assert!((last_visual_position - 11.79).abs() < 1e-5);

        let mut first_ack = TerminalSnapshot::empty(TerminalId::new(1), TerminalSize::new(8, 4));
        first_ack.viewport_position = 11;
        first_ack.rows_before = (0..8)
            .map(|_| Arc::from(vec![TerminalCell::default(); 8].into_boxed_slice()))
            .collect();
        first_ack.rows_after = (0..8)
            .map(|_| Arc::from(vec![TerminalCell::default(); 8].into_boxed_slice()))
            .collect();
        reconcile_visual_scroll(&mut state, &first_ack);
        let after_first_ack = state.observed_viewport_position as f32 + state.visual_unacked_rows;
        assert!((after_first_ack - last_visual_position).abs() < f32::EPSILON);

        for delta in [0.33_f32, 0.44] {
            let (_, repaint) = state.accumulate(delta);
            let visual_position =
                state.observed_viewport_position as f32 + state.visual_unacked_rows;
            assert!(repaint);
            assert!(visual_position > last_visual_position);
            last_visual_position = visual_position;
        }

        let mut final_ack = first_ack;
        final_ack.viewport_position = 12;
        reconcile_visual_scroll(&mut state, &final_ack);
        let final_visual_position =
            state.observed_viewport_position as f32 + state.visual_unacked_rows;
        let input_total: f32 = [0.21, 0.32, 0.41, 0.37, 0.48, 0.33, 0.44].into_iter().sum();
        assert!((final_visual_position - last_visual_position).abs() < f32::EPSILON);
        assert!((final_visual_position - (10.0 + input_total)).abs() < 1e-5);
    }

    #[test]
    fn fractional_scroll_positions_nearest_overscan_rows_without_a_gap() {
        assert_eq!(terminal_row_position(-1, 0.25), -0.75);
        assert_eq!(terminal_row_position(0, 0.25), 0.25);
        assert_eq!(terminal_row_position(4, -0.25), 3.75);
        assert_eq!(terminal_row_position(5, -0.25), 4.75);
    }

    #[test]
    fn multi_row_visual_offsets_select_only_visible_source_rows() {
        assert_eq!(
            terminal_visible_source_rows(4, 1.25).collect::<Vec<_>>(),
            vec![-2, -1, 0, 1, 2]
        );
        assert_eq!(
            terminal_visible_source_rows(4, -3.4).collect::<Vec<_>>(),
            vec![3, 4, 5, 6, 7]
        );
        assert_eq!(
            terminal_visible_source_rows(4, 2.0).collect::<Vec<_>>(),
            vec![-2, -1, 0, 1]
        );
    }

    #[test]
    fn prepared_row_band_stays_bounded_inside_the_raw_reserve() {
        let visible = terminal_visible_source_rows(24, 12.4);
        let prepared = terminal_prepared_source_rows(visible.clone(), -32, 24 + 32);

        assert_eq!(prepared.start, visible.start - PREPARED_ROW_LOOKAHEAD);
        assert_eq!(prepared.end, visible.end + PREPARED_ROW_LOOKAHEAD);
        assert_eq!(
            prepared.len(),
            visible.len() + 2 * PREPARED_ROW_LOOKAHEAD as usize
        );
        assert!(prepared.len() < 24 + 2 * 32);
    }

    #[test]
    fn viewport_ack_reuses_the_same_physical_cached_row() {
        assert_eq!(previous_cached_source_row(0, 11, 10), Some(-1));
        assert_eq!(previous_cached_source_row(5, 11, 10), Some(4));
        assert_eq!(previous_cached_source_row(-2, 9, 10), Some(-1));
    }

    #[test]
    fn cached_rows_rebuild_only_where_the_focused_cursor_was_or_is() {
        assert!(!terminal_cached_row_cursor_compatible(
            Some((2, 7)),
            Some((3, 0)),
            2,
            2,
        ));
        assert!(!terminal_cached_row_cursor_compatible(
            Some((2, 7)),
            Some((3, 0)),
            3,
            3,
        ));
        assert!(terminal_cached_row_cursor_compatible(
            Some((2, 7)),
            Some((3, 0)),
            1,
            1,
        ));
        assert!(!terminal_cached_row_cursor_compatible(
            Some((4, 5)),
            Some((4, 6)),
            4,
            4,
        ));
        assert!(terminal_cached_row_cursor_compatible(None, None, 4, 4));
    }

    #[test]
    fn terminal_input_preserves_printable_text_and_control_bytes() {
        assert!(terminal_key_uses_text_input_handler(&keystroke(
            "a",
            Some("a"),
            Modifiers::none(),
        )));
        assert!(!terminal_key_uses_text_input_handler(&keystroke(
            "enter",
            Some("\r"),
            Modifiers::none(),
        )));
        assert!(!terminal_key_uses_text_input_handler(&keystroke(
            "return",
            None,
            Modifiers::none(),
        )));
        assert_eq!(
            terminal_input_for_keystroke_with_modes(
                &keystroke("a", Some("a"), Modifiers::none()),
                TerminalModes::default(),
            ),
            Some("a".to_owned())
        );
        assert_eq!(
            terminal_input_for_keystroke_with_modes(
                &keystroke("c", Some("c"), Modifiers::control()),
                TerminalModes::default(),
            ),
            Some("\u{3}".to_owned())
        );
        assert_eq!(
            terminal_input_for_keystroke_with_modes(
                &keystroke("backspace", None, Modifiers::none()),
                TerminalModes::default(),
            ),
            Some("\u{7f}".to_owned())
        );
        assert_eq!(
            terminal_input_for_keystroke_with_modes(
                &keystroke("enter", None, Modifiers::none()),
                TerminalModes::default(),
            ),
            Some("\r".to_owned())
        );
    }

    #[test]
    fn terminal_home_and_end_use_standard_and_application_sequences() {
        let normal = TerminalModes::default();
        assert_eq!(
            terminal_input_for_keystroke_with_modes(
                &keystroke("home", None, Modifiers::none()),
                normal,
            ),
            Some("\u{1b}[H".to_owned())
        );
        assert_eq!(
            terminal_input_for_keystroke_with_modes(
                &keystroke("end", None, Modifiers::none()),
                normal,
            ),
            Some("\u{1b}[F".to_owned())
        );

        let application = TerminalModes {
            application_cursor: true,
            ..normal
        };
        assert_eq!(
            terminal_input_for_keystroke_with_modes(
                &keystroke("home", None, Modifiers::none()),
                application,
            ),
            Some("\u{1b}OH".to_owned())
        );
        assert_eq!(
            terminal_input_for_keystroke_with_modes(
                &keystroke("end", None, Modifiers::none()),
                application,
            ),
            Some("\u{1b}OF".to_owned())
        );
        assert_eq!(
            terminal_input_for_keystroke_with_modes(
                &keystroke("home", None, Modifiers::shift()),
                application,
            ),
            Some("\u{1b}[1;2H".to_owned())
        );
    }

    #[test]
    fn terminal_selection_extracts_text_without_wide_spacers_or_padding() {
        let terminal_id = TerminalId::new(1);
        let mut snapshot = TerminalSnapshot::empty(terminal_id, TerminalSize::new(8, 2));
        for (column, character) in "hello".chars().enumerate() {
            snapshot.cell_mut(0, column).unwrap().character = character;
        }
        for (column, character) in "world".chars().enumerate() {
            snapshot.cell_mut(1, column).unwrap().character = character;
        }
        snapshot.cell_mut(0, 5).unwrap().flags.set_wide_spacer(true);
        snapshot.cell_mut(0, 4).unwrap().character = '界';
        snapshot.cell_mut(0, 4).unwrap().flags.set_wide(true);
        let selection = TerminalSelection {
            terminal_id,
            anchor: endpoint(0, 0, TerminalSelectionSide::Left),
            head: endpoint(1, 7, TerminalSelectionSide::Right),
        };
        assert_eq!(
            selected_terminal_text(&snapshot, selection),
            "hell界\nworld"
        );
    }

    #[test]
    fn terminal_selection_includes_painted_scrollback_rows() {
        let terminal_id = TerminalId::new(1);
        let mut snapshot = TerminalSnapshot::empty(terminal_id, TerminalSize::new(4, 2));
        let history_row = "hist"
            .chars()
            .map(|character| TerminalCell {
                character,
                ..TerminalCell::default()
            })
            .collect::<Vec<_>>();
        snapshot
            .rows_before
            .push(Arc::from(history_row.into_boxed_slice()));
        let selection = TerminalSelection {
            terminal_id,
            anchor: endpoint(-1, 0, TerminalSelectionSide::Left),
            head: endpoint(-1, 3, TerminalSelectionSide::Right),
        };

        let (start, end) = selection_bounds(&snapshot, selection).unwrap();
        assert_eq!((start.row, end.row), (-1, -1));
        assert_eq!(selected_terminal_text(&snapshot, selection), "hist");
    }

    #[test]
    fn shift_click_extends_the_previous_selection_anchor() {
        let terminal_id = TerminalId::new(1);
        let other_terminal_id = TerminalId::new(2);
        let previous = TerminalSelection {
            terminal_id,
            anchor: endpoint(1, 2, TerminalSelectionSide::Left),
            head: endpoint(2, 3, TerminalSelectionSide::Right),
        };
        let extended = terminal_selection_after_click(
            Some(previous),
            terminal_id,
            endpoint(5, 1, TerminalSelectionSide::Left),
            true,
        );
        assert_eq!(extended.anchor, previous.anchor);
        assert_eq!(
            extended.head.position,
            TerminalCellPosition { row: 5, column: 1 }
        );

        let new_selection = terminal_selection_after_click(
            Some(previous),
            other_terminal_id,
            endpoint(4, 0, TerminalSelectionSide::Left),
            true,
        );
        assert_eq!(new_selection.anchor, new_selection.head);
        assert_eq!(new_selection.terminal_id, other_terminal_id);
    }

    #[test]
    fn selection_autoscroll_direction_is_edge_triggered() {
        assert_eq!(terminal_selection_autoscroll_direction(4.0, 200.0), Some(3));
        assert_eq!(
            terminal_selection_autoscroll_direction(198.0, 200.0),
            Some(-3)
        );
        assert_eq!(terminal_selection_autoscroll_direction(100.0, 200.0), None);
        assert_eq!(terminal_selection_autoscroll_direction(0.0, 0.0), None);
    }

    #[test]
    fn selection_sides_choose_only_fully_covered_cells() {
        let terminal_id = TerminalId::new(1);
        let mut snapshot = TerminalSnapshot::empty(terminal_id, TerminalSize::new(4, 1));
        for (column, character) in "abcd".chars().enumerate() {
            snapshot.cell_mut(0, column).unwrap().character = character;
        }

        let first_cell = TerminalSelection {
            terminal_id,
            anchor: endpoint(0, 0, TerminalSelectionSide::Left),
            head: endpoint(0, 1, TerminalSelectionSide::Left),
        };
        assert_eq!(selection_bounds(&snapshot, first_cell).unwrap().1.column, 0);

        let boundary_only = TerminalSelection {
            terminal_id,
            anchor: endpoint(0, 0, TerminalSelectionSide::Right),
            head: endpoint(0, 1, TerminalSelectionSide::Left),
        };
        assert!(selection_bounds(&snapshot, boundary_only).is_none());

        let second_cell = TerminalSelection {
            terminal_id,
            anchor: endpoint(0, 0, TerminalSelectionSide::Right),
            head: endpoint(0, 1, TerminalSelectionSide::Right),
        };
        let (start, end) = selection_bounds(&snapshot, second_cell).unwrap();
        assert_eq!((start.column, end.column), (1, 1));
    }

    #[test]
    fn selection_follows_user_viewport_scroll() {
        let terminal_id = TerminalId::new(1);
        let mut previous = TerminalSnapshot::empty(terminal_id, TerminalSize::new(8, 4));
        previous.display_offset = 2;
        previous.viewport_position = 2;
        let mut next = previous.clone();
        next.display_offset = 6;
        next.viewport_position = 6;

        let selection = TerminalSelection {
            terminal_id,
            anchor: endpoint(1, 2, TerminalSelectionSide::Left),
            head: endpoint(2, 4, TerminalSelectionSide::Right),
        };
        let mut wrapped = Some(selection);
        shift_selection_for_viewport(&mut wrapped, terminal_id, &previous, &next);
        let selection = wrapped.expect("selection remains present");

        assert_eq!(selection.anchor.position.row, 5);
        assert_eq!(selection.head.position.row, 6);
    }

    #[test]
    fn output_growth_does_not_move_selection_in_a_pinned_viewport() {
        let terminal_id = TerminalId::new(1);
        let mut previous = TerminalSnapshot::empty(terminal_id, TerminalSize::new(8, 4));
        previous.display_offset = 2;
        previous.viewport_position = 2;
        let mut next = previous.clone();
        next.display_offset = 6;
        // Output can increase display_offset while the pinned viewport stays
        // on the same visible cells. It must not move the selection highlight.
        next.viewport_position = previous.viewport_position;

        let selection = TerminalSelection {
            terminal_id,
            anchor: endpoint(1, 2, TerminalSelectionSide::Left),
            head: endpoint(2, 4, TerminalSelectionSide::Right),
        };
        let mut wrapped = Some(selection);
        shift_selection_for_viewport(&mut wrapped, terminal_id, &previous, &next);
        let selection = wrapped.expect("selection remains present");

        assert_eq!(selection.anchor.position.row, 1);
        assert_eq!(selection.head.position.row, 2);
    }

    #[test]
    fn paint_downward_offset_is_bounded_by_the_semantic_live_bottom() {
        // Regression: browsing history with the live tail inside the 32-row
        // overscan, an input-triggered jump to the live bottom used to be
        // painted clamped to -rows_after.len() (a visible upward jump) until
        // the ScrollTo(0) ACK landed. The clamp mirrors
        // terminal_scroll_offset_for_snapshot.
        let terminal_id = TerminalId::new(9);
        let mut snapshot = TerminalSnapshot::empty(terminal_id, TerminalSize::new(8, 5));
        snapshot.viewport_position = 2;
        snapshot.display_offset = 2;
        // Simulate a real from_term snapshot: last_source_row is the
        // distance from the viewport bottom to the newest materialized
        // row (the live tail when the overscan reaches it).
        snapshot.last_source_row = 2;
        snapshot.rows_before.push(Arc::from(vec![TerminalCell::default(); 8].into_boxed_slice()));
        snapshot.rows_before.push(Arc::from(vec![TerminalCell::default(); 8].into_boxed_slice()));
        snapshot.rows_after.push(Arc::from(vec![TerminalCell::default(); 8].into_boxed_slice()));
        snapshot.rows_after.push(Arc::from(vec![TerminalCell::default(); 8].into_boxed_slice()));

        // The clamp (mirroring terminal_scroll_offset_for_snapshot) allows
        // the full -viewport_position jump, not just -rows_after.len().
        let clamp = |offset: f32, snap: &TerminalSnapshot| {
            let overscan = snap.rows_before.len() as f32;
            let limit = snap.last_source_row.min(overscan as i64) as f32;
            offset.clamp(-limit, overscan)
        };
        let jump = -2.0;
        // With the materialized window reaching the live tail (last_source_row=2),
        // the full jump to the live bottom is allowed.
        assert_eq!(clamp(jump, &snapshot), -2.0);

        // If the materialized window stops one row short of the live tail
        // while browsing deeper (viewport 3), the paint offset is capped
        // at the materialized live bottom (-2) instead of pretending to
        // address the tail row (-3).
        let short = TerminalSnapshot {
            last_source_row: 1,
            viewport_position: 3,
            ..snapshot.clone()
        };
        assert_eq!(clamp(-3.0, &short), -1.0);

        // When the materialized window already stops at the live tail
        // (no overscan below), the jump is still bounded by the
        // materialized live bottom.
        let bottom = TerminalSnapshot {
            last_source_row: 2,
            rows_after: Vec::new(),
            ..snapshot.clone()
        };
        assert_eq!(clamp(jump, &bottom), -2.0);

        // Deep browsing: the user is 10 rows above the live tail and the
        // materialized window stops 2 rows short of it. The input-triggered
        // jump must be capped at the materialized live bottom (-2), not
        // pretend to address the tail row (-10).
        let deep = TerminalSnapshot {
            last_source_row: 2,
            viewport_position: 10,
            ..snapshot.clone()
        };
        assert_eq!(clamp(-10.0, &deep), -2.0);
        // With a larger materialized window (overscan 4) the jump is
        // capped at the materialized live bottom (-4), still short of the
        // semantic tail (-10).
        let deep_reached = TerminalSnapshot {
            last_source_row: 4,
            viewport_position: 10,
            ..snapshot.clone()
        };
        assert_eq!(clamp(-10.0, &deep_reached), -2.0);
    }

    #[test]
    fn wide_character_selection_expands_to_both_grid_cells() {
        let terminal_id = TerminalId::new(1);
        let mut snapshot = TerminalSnapshot::empty(terminal_id, TerminalSize::new(5, 1));
        snapshot.cell_mut(0, 1).unwrap().character = '界';
        snapshot.cell_mut(0, 1).unwrap().flags.set_wide(true);
        snapshot.cell_mut(0, 2).unwrap().flags.set_wide_spacer(true);
        let selection = TerminalSelection {
            terminal_id,
            anchor: endpoint(0, 0, TerminalSelectionSide::Left),
            head: endpoint(0, 1, TerminalSelectionSide::Right),
        };

        let (_, end) = selection_bounds(&snapshot, selection).unwrap();
        assert_eq!(end.column, 2);
        assert!(is_terminal_cell_selected(
            &snapshot,
            Some(selection),
            terminal_id,
            0,
            1,
        ));
        assert!(is_terminal_cell_selected(
            &snapshot,
            Some(selection),
            terminal_id,
            0,
            2,
        ));

        let half_cell_drag = TerminalSelection {
            terminal_id,
            anchor: endpoint(0, 1, TerminalSelectionSide::Left),
            head: endpoint(0, 2, TerminalSelectionSide::Right),
        };
        assert!(is_terminal_cell_selected(
            &snapshot,
            Some(half_cell_drag),
            terminal_id,
            0,
            1,
        ));
        assert!(is_terminal_cell_selected(
            &snapshot,
            Some(half_cell_drag),
            terminal_id,
            0,
            2,
        ));
    }

    #[test]
    fn leading_wide_placeholders_do_not_become_a_second_character() {
        let terminal_id = TerminalId::new(1);
        let mut snapshot = TerminalSnapshot::empty(terminal_id, TerminalSize::new(4, 2));
        snapshot
            .cell_mut(0, 3)
            .unwrap()
            .flags
            .set_leading_wide_spacer(true);
        snapshot.cell_mut(1, 0).unwrap().character = '界';
        snapshot.cell_mut(1, 0).unwrap().flags.set_wide(true);
        snapshot.cell_mut(1, 1).unwrap().flags.set_wide_spacer(true);

        let wrapped_wide_character = TerminalSelection {
            terminal_id,
            anchor: endpoint(0, 3, TerminalSelectionSide::Left),
            head: endpoint(1, 0, TerminalSelectionSide::Right),
        };
        let (start, end) = selection_bounds(&snapshot, wrapped_wide_character).unwrap();
        assert_eq!((start.row, start.column), (1, 0));
        assert_eq!((end.row, end.column), (1, 1));
        assert_eq!(
            selected_terminal_text(&snapshot, wrapped_wide_character),
            "界"
        );

        let placeholder_only = TerminalSelection {
            terminal_id,
            anchor: endpoint(0, 3, TerminalSelectionSide::Left),
            head: endpoint(0, 3, TerminalSelectionSide::Right),
        };
        assert!(selection_bounds(&snapshot, placeholder_only).is_none());
    }

    #[test]
    fn terminal_rows_shape_wide_cells_with_two_cell_advances() {
        let terminal_id = TerminalId::new(1);
        let mut snapshot = TerminalSnapshot::empty(terminal_id, TerminalSize::new(5, 1));
        snapshot.cell_mut(0, 0).unwrap().character = 'a';
        snapshot.cell_mut(0, 1).unwrap().character = '界';
        snapshot.cell_mut(0, 1).unwrap().flags.set_wide(true);
        snapshot.cell_mut(0, 2).unwrap().flags.set_wide_spacer(true);
        snapshot.cell_mut(0, 3).unwrap().character = 'b';
        let options = TerminalRenderOptions {
            metrics: TerminalMetrics::default(),
            theme: ThemeColors {
                terminal_background: 0,
                terminal_foreground: 1,
                selection_background: 2,
                cursor_foreground: 3,
                cursor_background: 4,
                inactive_cursor: 5,
                inverse_foreground: 6,
                inverse_background: 7,
                pane_background: 8,
                active_pane_border: 9,
                inactive_pane_border: 10,
                accent: 9,
                accent_foreground: 11,
                chrome_background: 11,
                tab_active_background: 12,
                tab_inactive_background: 13,
                tab_add_background: 14,
                ui_foreground: 15,
                sidebar_background: 16,
                sidebar_connection_background: 16,
                sidebar_connection_active_background: 16,
                sidebar_workspace_background: 17,
                sidebar_agent_background: 18,
                sidebar_drag_indicator: 19,
                sidebar_workspace_active_background: 19,
                sidebar_agent_active_background: 20,
                agent_colors: [21; 13],
            },
            cursor_focused: false,
            scroll_offset_rows: 0.0,
        };
        let (chunks, _) =
            terminal_row_data(&snapshot, 0, None, options, "Sarasa Term SC Nerd Font");
        assert_eq!(
            chunks
                .iter()
                .map(|chunk| (chunk.start_column, chunk.width_columns, chunk.text.as_str()))
                .collect::<Vec<_>>(),
            vec![(0, 1, "a"), (1, 2, "界"), (3, 1, "b ")]
        );
    }

    #[test]
    fn isolated_unicode_does_not_force_ascii_neighbors_into_per_cell_shaping() {
        let terminal_id = TerminalId::new(1);
        let mut snapshot = TerminalSnapshot::empty(terminal_id, TerminalSize::new(6, 1));
        for (column, character) in ['a', 'b', '·', 'c', 'd'].into_iter().enumerate() {
            snapshot.cell_mut(0, column).unwrap().character = character;
        }
        let options = TerminalRenderOptions {
            metrics: TerminalMetrics::default(),
            theme: AppConfig::default().theme.colors(),
            cursor_focused: false,
            scroll_offset_rows: 0.0,
        };

        let (chunks, _) = terminal_row_data(&snapshot, 0, None, options, "monospace");
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].text, "ab");
        assert!(!chunks[0].requires_cell_scaling);
        assert_eq!(chunks[1].text, "·");
        assert!(chunks[1].requires_cell_scaling);
        assert_eq!(chunks[2].text, "cd ");
        assert!(!chunks[2].requires_cell_scaling);
    }

    #[test]
    fn dim_terminal_colors_match_alacritty_intensity() {
        assert_eq!(dim_terminal_color(0xffffff), 0xa8a8a8);
        assert_eq!(dim_terminal_color(0x804020), 0x542a15);
    }

    #[test]
    fn oversized_unicode_uses_a_bounded_font_scale() {
        assert!((terminal_fit_scale(21.0, 14.0) - (2.0 / 3.0)).abs() < 1e-6);
        assert_eq!(terminal_fit_scale(14.0, 14.0), 1.0);
        assert_eq!(terminal_fit_scale(12.0, 14.0), 1.0);
        assert_eq!(terminal_fit_scale(f32::INFINITY, 14.0), 1.0);
    }

    #[test]
    fn terminal_grid_geometry_is_shared_by_bounds_and_mouse_hit_testing() {
        let metrics = TerminalMetrics {
            cell_width: 8.4,
            line_height: 17.3,
            scale_factor: 2.0,
        };
        let bounds = Bounds::new(point(px(0.3), px(1.2)), size(px(42.0), px(35.0)));
        let edges: Vec<_> = (0..8)
            .map(|column| {
                terminal_grid_edge(
                    f32::from(bounds.origin.x),
                    metrics.cell_width,
                    column,
                    metrics.scale_factor,
                )
            })
            .collect();
        assert!(edges.windows(2).all(|pair| pair[0] <= pair[1]));

        let first = terminal_cell_bounds(bounds, metrics, 0, 0, 1);
        let second = terminal_cell_bounds(bounds, metrics, 0, 1, 1);
        let first_x = f32::from(first.origin.x) + f32::from(first.size.width) * 0.75;
        let second_x = f32::from(second.origin.x) + f32::from(second.size.width) * 0.25;
        let y = f32::from(first.origin.y) + f32::from(first.size.height) * 0.5;

        let first_hit = terminal_mouse_position(point(px(first_x), px(y)), Some(bounds), metrics);
        assert_eq!(
            (first_hit.column, first_hit.side),
            (1, TerminalSelectionSide::Right)
        );
        let second_hit = terminal_mouse_position(point(px(second_x), px(y)), Some(bounds), metrics);
        assert_eq!(
            (second_hit.column, second_hit.side),
            (2, TerminalSelectionSide::Left)
        );
    }

    #[test]
    fn utf16_ime_offsets_follow_unicode_scalar_boundaries() {
        let text = "a界😀";
        assert_eq!(utf16_len(text), 4);
        assert_eq!(utf16_offset_to_byte(text, 0), 0);
        assert_eq!(utf16_offset_to_byte(text, 1), 1);
        assert_eq!(utf16_offset_to_byte(text, 2), 4);
        assert_eq!(utf16_offset_to_byte(text, 3), 4);
        assert_eq!(utf16_offset_to_byte(text, 4), text.len());
        assert_eq!(normalize_utf16_range(0..99, 4), 0..4);
        assert_eq!(
            normalize_utf16_range(std::ops::Range { start: 3, end: 1 }, 4),
            1..1
        );
    }

    #[gpui::test]
    fn windows_switch_shared_connection_projections_without_cross_talk(
        cx: &mut gpui::TestAppContext,
    ) {
        let mut local_host = crate::app::ModelHost::start();
        let local_client: Arc<dyn CommandTransport> = Arc::new(local_host.client());
        let local_operation = local_client
            .dispatch(AppCommand::Workspace(WorkspaceCommand::Create))
            .unwrap();
        local_client.wait_operation(local_operation).unwrap();

        let mut remote_host = crate::app::ModelHost::start();
        let remote_client: Arc<dyn CommandTransport> = Arc::new(remote_host.client());
        for _ in 0..2 {
            let operation = remote_client
                .dispatch(AppCommand::Workspace(WorkspaceCommand::Create))
                .unwrap();
            remote_client.wait_operation(operation).unwrap();
        }

        let local_id = ConnectionId::new(1);
        let remote_id = ConnectionId::new(2);
        let connections = vec![
            WorkspaceConnection {
                id: local_id,
                title: "Local".to_owned(),
                kind: WorkspaceConnectionKind::Local,
                client: local_client,
                snapshot: local_host.client().state_dump().unwrap(),
            },
            WorkspaceConnection {
                id: remote_id,
                title: "build-box".to_owned(),
                kind: WorkspaceConnectionKind::Remote,
                client: remote_client,
                snapshot: remote_host.client().state_dump().unwrap(),
            },
        ];
        let first_connections = connections.clone();
        let (view, cx) = cx.add_window_view(move |_, cx| {
            WorkspaceView::new_with_connections(
                None,
                first_connections,
                local_id,
                cx.focus_handle(),
                AppConfig::default(),
            )
        });
        let (other_view, cx) = cx.add_window_view(move |_, cx| {
            WorkspaceView::new_with_connections(
                None,
                connections,
                local_id,
                cx.focus_handle(),
                AppConfig::default(),
            )
        });
        assert_eq!(
            view.update_in(cx, |view, _, _| (
                view.active_connection,
                view.workspace_dumps().len()
            )),
            (local_id, 1)
        );
        view.update_in(cx, |view, _, cx| {
            assert!(view.select_connection_locally(remote_id, cx));
        });
        assert_eq!(
            view.update_in(cx, |view, _, _| (
                view.active_connection,
                view.workspace_dumps().len()
            )),
            (remote_id, 2)
        );

        assert_eq!(
            other_view.update_in(cx, |view, _, _| (
                view.active_connection,
                view.workspace_dumps().len()
            )),
            (local_id, 1),
            "switching one window must not change another window's projection"
        );
        other_view.update_in(cx, |view, _, cx| {
            assert!(view.select_connection_locally(remote_id, cx));
        });
        view.update_in(cx, |view, _, cx| {
            assert!(view.select_connection_locally(local_id, cx));
        });
        assert_eq!(
            view.update_in(cx, |view, _, _| view.active_connection),
            local_id
        );
        assert_eq!(
            other_view.update_in(cx, |view, _, _| view.active_connection),
            remote_id
        );

        view.update_in(cx, |view, _, cx| view.remove_connection(remote_id, cx));
        assert_eq!(
            view.update_in(cx, |view, _, _| (
                view.active_connection,
                view.workspace_dumps().len()
            )),
            (local_id, 1)
        );

        local_host.shutdown();
        remote_host.shutdown();
    }

    #[gpui::test]
    fn ime_composition_state_commits_once_to_the_focused_terminal(cx: &mut gpui::TestAppContext) {
        let mut host = crate::app::ModelHost::start();
        let client = std::sync::Arc::new(host.client());
        let workspace = client
            .dispatch(crate::command::AppCommand::Workspace(
                crate::command::WorkspaceCommand::Create,
            ))
            .unwrap();
        client.wait_operation(workspace).unwrap();
        let tab = client
            .dispatch(crate::command::AppCommand::Tab(
                crate::command::TabCommand::New { title: None },
            ))
            .unwrap();
        client.wait_operation(tab).unwrap();
        let program = crate::terminal::default_shell_program();
        let spawn = client
            .dispatch(crate::command::AppCommand::Terminal(
                crate::command::TerminalCommand::Spawn {
                    pane_id: None,
                    args: crate::terminal::default_shell_args(&program),
                    program,
                    columns: 80,
                    lines: 24,
                },
            ))
            .unwrap();
        let spawn = client.wait_operation(spawn).unwrap();
        let terminal_id = match spawn.result.unwrap() {
            crate::command::OperationResult::TerminalSpawned { terminal_id } => terminal_id,
            result => panic!("unexpected result: {result:?}"),
        };
        let snapshot = client.state_dump().unwrap();
        let (view, cx) = cx.add_window_view(|_, cx| {
            WorkspaceView::new(client.clone(), snapshot.clone(), cx.focus_handle())
        });
        view.update_in(cx, |view, window, cx| {
            window.focus(&view.focus_handle, cx);
            view.ime_terminal = Some(terminal_id);
            <WorkspaceView as EntityInputHandler>::replace_and_mark_text_in_range(
                view,
                None,
                "拼音",
                Some(2..2),
                window,
                cx,
            );
        });
        let (marked_text, selected_range) = view.update_in(cx, |view, _, _| {
            (
                view.ime_marked_text.clone(),
                view.ime_selected_range.clone(),
            )
        });
        assert_eq!(marked_text, "拼音");
        assert_eq!(selected_range, 2..2);

        view.update_in(cx, |view, window, cx| {
            <WorkspaceView as EntityInputHandler>::replace_text_in_range(
                view,
                None,
                "print -r -- IME_中",
                window,
                cx,
            );
        });
        assert!(view.update_in(cx, |view, _, _| view.ime_marked_text.is_empty()));
        cx.simulate_keystrokes("enter");
        client
            .terminal_contains(terminal_id, "IME_中", std::time::Duration::from_secs(5))
            .unwrap();
        host.shutdown();
    }

    #[test]
    fn terminal_input_honors_application_cursor_and_mouse_reporting_modes() {
        assert_eq!(
            terminal_input_for_keystroke_with_modes(
                &keystroke("up", None, Modifiers::none()),
                TerminalModes {
                    application_cursor: true,
                    ..TerminalModes::default()
                },
            ),
            Some("\u{1b}OA".to_owned())
        );
        let event = ScrollWheelEvent {
            delta: ScrollDelta::Lines(Point { x: 0.0, y: 1.0 }),
            ..ScrollWheelEvent::default()
        };
        assert_eq!(
            terminal_mouse_input(
                &event,
                TerminalMouseContext {
                    modes: TerminalModes {
                        mouse_reporting: true,
                        sgr_mouse: true,
                        ..TerminalModes::default()
                    },
                    bounds: None,
                    metrics: TerminalMetrics::default(),
                },
            ),
            Some(b"\x1b[<64;1;1M".to_vec())
        );
    }

    #[test]
    fn tab_bar_edge_indicators_follow_scroll_position() {
        // No overflow: never show either edge.
        assert_eq!(tab_bar_edge_indicators(0.0, 0.0), (false, false));
        assert_eq!(tab_bar_edge_indicators(0.8, -0.5), (false, false));
        // Overflow at the leading edge: only the right side has content.
        assert_eq!(tab_bar_edge_indicators(200.0, 0.0), (false, true));
        // Mid-scroll: both sides hidden content.
        assert_eq!(tab_bar_edge_indicators(200.0, -100.0), (true, true));
        // Fully scrolled to the end: only the left side has hidden content.
        assert_eq!(tab_bar_edge_indicators(200.0, -200.0), (true, false));
    }

    #[test]
    fn sidebar_drop_boundaries_reject_group_interiors_and_out_of_range_points() {
        let groups = [
            SidebarGroupGeometry {
                top: 10.0,
                bottom: 38.0,
            },
            SidebarGroupGeometry {
                top: 38.0,
                bottom: 84.0,
            },
            SidebarGroupGeometry {
                top: 84.0,
                bottom: 112.0,
            },
        ];
        assert_eq!(
            sidebar_drop_boundary(11.0, &groups, 10.0, 112.0, 14.0),
            Some((0, 10.0))
        );
        assert_eq!(
            sidebar_drop_boundary(39.0, &groups, 10.0, 112.0, 14.0),
            Some((1, 38.0))
        );
        assert_eq!(
            sidebar_drop_boundary(111.0, &groups, 10.0, 112.0, 14.0),
            Some((3, 112.0))
        );
        assert_eq!(
            sidebar_drop_boundary(60.0, &groups, 10.0, 112.0, 14.0),
            None
        );
        assert_eq!(sidebar_drop_boundary(0.0, &groups, 10.0, 112.0, 14.0), None);
        assert_eq!(
            sidebar_drop_boundary(125.0, &groups, 10.0, 112.0, 14.0),
            None
        );
    }

    #[test]
    fn sidebar_reorder_index_uses_after_removal_semantics() {
        assert_eq!(sidebar_reorder_final_index(0, 0, 3), 0);
        assert_eq!(sidebar_reorder_final_index(0, 3, 3), 2);
        assert_eq!(sidebar_reorder_final_index(1, 0, 3), 0);
        assert_eq!(sidebar_reorder_final_index(1, 1, 3), 1);
        assert_eq!(sidebar_reorder_final_index(1, 2, 3), 1);
        assert_eq!(sidebar_reorder_final_index(2, 0, 3), 0);
        assert_eq!(sidebar_reorder_final_index(2, 3, 3), 2);
        assert_eq!(sidebar_reorder_final_index(99, 99, 3), 0);
    }

    #[test]
    fn split_ratio_geometry_clamps_pointer_to_dispatcher_range() {
        assert_eq!(split_ratio_for_pointer(100.0, 0.0, 200.0, 0.0), 0.5);
        assert_eq!(split_ratio_for_pointer(-20.0, 0.0, 200.0, 0.0), 0.05);
        assert_eq!(split_ratio_for_pointer(240.0, 0.0, 200.0, 0.0), 0.95);
        assert_eq!(split_ratio_for_pointer(20.0, 10.0, 0.0, 0.0), 0.5);
        assert_eq!(split_ratio_for_pointer(f32::NAN, 0.0, 200.0, 0.0), 0.5);
    }

    #[test]
    fn split_ratio_accounts_for_the_fixed_divider() {
        // The flex renderer gives children (extent - divider) to share and
        // centers the fixed divider on the boundary; the pointer maps back
        // with the same geometry so the committed ratio is what the user
        // dragged to, without midpoint bias.
        assert_eq!(split_ratio_for_pointer(33.0, 0.0, 106.0, 6.0), 0.3);
        assert_eq!(split_ratio_for_pointer(53.0, 0.0, 106.0, 6.0), 0.5);
        assert_eq!(split_ratio_for_pointer(93.0, 0.0, 106.0, 6.0), 0.9);
        // No usable space without the divider band itself.
        assert_eq!(split_ratio_for_pointer(50.0, 0.0, 6.0, 6.0), 0.5);
    }

    #[test]
    fn sidebar_color_helpers_mix_and_measure() {
        assert_eq!(mix_rgb(0xff0000, 0x0000ff, 1.0), 0xff0000);
        assert_eq!(mix_rgb(0xff0000, 0x0000ff, 0.0), 0x0000ff);
        assert_eq!(mix_rgb(0x808080, 0x000000, 0.5), 0x404040);
        assert_eq!(channel_distance(0x102030, 0x102030), 0);
        assert_eq!(channel_distance(0x00ff00, 0x000000), 0xff);
        assert_eq!(channel_distance(0x0a141e, 0x000000), 0x1e);
    }

    #[test]
    fn terminal_input_encodes_navigation_and_alt_sequences() {
        assert_eq!(
            terminal_input_for_keystroke_with_modes(
                &keystroke("up", None, Modifiers::none()),
                TerminalModes::default(),
            ),
            Some("\u{1b}[A".to_owned())
        );
        assert_eq!(
            terminal_input_for_keystroke_with_modes(
                &keystroke(
                    "left",
                    None,
                    Modifiers {
                        control: true,
                        ..Modifiers::none()
                    },
                ),
                TerminalModes::default(),
            ),
            Some("\u{1b}[1;5D".to_owned())
        );
        assert_eq!(
            terminal_input_for_keystroke_with_modes(
                &keystroke(
                    "x",
                    Some("x"),
                    Modifiers {
                        alt: true,
                        ..Modifiers::none()
                    },
                ),
                TerminalModes::default(),
            ),
            Some("\u{1b}x".to_owned())
        );
        assert_eq!(
            terminal_input_for_keystroke_with_modes(
                &keystroke(
                    "tab",
                    None,
                    Modifiers {
                        shift: true,
                        ..Modifiers::none()
                    },
                ),
                TerminalModes::default(),
            ),
            Some("\u{1b}[Z".to_owned())
        );
    }

    #[gpui::test]
    fn focused_workspace_routes_keystrokes_to_real_zsh(cx: &mut gpui::TestAppContext) {
        let mut host = crate::app::ModelHost::start();
        let client = std::sync::Arc::new(host.client());
        let workspace = client
            .dispatch(crate::command::AppCommand::Workspace(
                crate::command::WorkspaceCommand::Create,
            ))
            .unwrap();
        assert_eq!(
            client.wait_operation(workspace).unwrap().status,
            crate::command::OperationStatus::Succeeded
        );
        let tab = client
            .dispatch(crate::command::AppCommand::Tab(
                crate::command::TabCommand::New { title: None },
            ))
            .unwrap();
        assert_eq!(
            client.wait_operation(tab).unwrap().status,
            crate::command::OperationStatus::Succeeded
        );
        let program = crate::terminal::default_shell_program();
        let spawn = client
            .dispatch(crate::command::AppCommand::Terminal(
                crate::command::TerminalCommand::Spawn {
                    pane_id: None,
                    args: crate::terminal::default_shell_args(&program),
                    program,
                    columns: 80,
                    lines: 24,
                },
            ))
            .unwrap();
        let spawn = client.wait_operation(spawn).unwrap();
        let _initial_terminal_id = match spawn.result.unwrap() {
            crate::command::OperationResult::TerminalSpawned { terminal_id } => terminal_id,
            result => panic!("unexpected result: {result:?}"),
        };
        let snapshot = client.state_dump().unwrap();
        cx.update(|cx| cx.bind_keys(crate::ui::application::window_key_bindings()));
        let (view, cx) = cx.add_window_view(|_, cx| {
            WorkspaceView::new(client.clone(), snapshot.clone(), cx.focus_handle())
        });
        view.update_in(cx, |view, window, cx| {
            window.focus(&view.focus_handle, cx);
        });
        cx.simulate_keystrokes("cmd-\\");
        let split_state = client.state_dump().unwrap();
        let split_tree = &split_state.workspace.as_ref().unwrap().tabs[0].tree;
        assert_eq!(split_tree.pane_count(), 2);
        assert_tree_is_terminal(split_tree);

        cx.simulate_keystrokes("cmd--");
        let vertical_split_state = client.state_dump().unwrap();
        let vertical_split_tab = &vertical_split_state.workspace.as_ref().unwrap().tabs[0];
        let vertical_split_tree = &vertical_split_tab.tree;
        assert_eq!(vertical_split_tree.pane_count(), 3);
        assert_tree_is_terminal(vertical_split_tree);
        let active_terminal_id =
            terminal_id_for_pane(vertical_split_tree, vertical_split_tab.active_pane).unwrap();

        cx.simulate_input("print -r -- UI_KEYBOARD_READY");
        cx.simulate_keystrokes("enter");
        client
            .terminal_contains(
                active_terminal_id,
                "UI_KEYBOARD_READY",
                std::time::Duration::from_secs(5),
            )
            .unwrap();

        cx.simulate_keystrokes("cmd-t");
        let tab_state = client.state_dump().unwrap();
        let workspace = tab_state.workspace.as_ref().unwrap();
        assert_eq!(workspace.tabs.len(), 2);
        assert_tree_is_terminal(&workspace.tabs[1].tree);

        host.shutdown();
    }

    #[gpui::test]
    fn mouse_selection_begins_and_survives_reinstall(cx: &mut gpui::TestAppContext) {
        let mut host = crate::app::ModelHost::start();
        let client = std::sync::Arc::new(host.client());
        let workspace = client
            .dispatch(crate::command::AppCommand::Workspace(
                crate::command::WorkspaceCommand::Create,
            ))
            .unwrap();
        client.wait_operation(workspace).unwrap();
        let tab = client
            .dispatch(crate::command::AppCommand::Tab(
                crate::command::TabCommand::New { title: None },
            ))
            .unwrap();
        client.wait_operation(tab).unwrap();

        let dump = client.state_dump().unwrap();
        let terminal_id = displayed_terminal_id(&dump);
        client
            .dispatch(crate::command::AppCommand::Terminal(
                crate::command::TerminalCommand::SendText {
                    terminal_id: Some(terminal_id),
                    pane_id: None,
                    text: "echo MOUSE_SEL_A\n".to_owned(),
                },
            ))
            .unwrap();
        client
            .terminal_contains(
                terminal_id,
                "MOUSE_SEL_A",
                std::time::Duration::from_secs(5),
            )
            .unwrap();
        let dump = client.state_dump().unwrap();
        let local_snapshot = Arc::new(crate::terminal::snapshot_from_replay(
            &client.terminal_replay(terminal_id).unwrap(),
            2_000,
        ));

        let (view, cx) = cx.add_window_view(|_, cx| {
            WorkspaceView::new(client.clone(), dump.clone(), cx.focus_handle())
        });
        view.update_in(cx, |view, _window, cx| {
            view.terminal_snapshots
                .insert(terminal_id, local_snapshot.clone());
            // Give hit-testing a stable, measured grid (the real window
            // measures via request_layout; the test host does not paint).
            view.terminal_metrics = TerminalMetrics {
                cell_width: 8.0,
                line_height: 16.0,
                scale_factor: 1.0,
            };
            view.terminal_bounds
                .lock()
                .expect("bounds poisoned")
                .insert(
                    terminal_id,
                    Bounds::new(point(px(0.0), px(0.0)), size(px(640.0), px(384.0))),
                );
            view.begin_terminal_selection(terminal_id, point(px(12.0), px(20.0)), false, cx);
            let selection = view
                .selection
                .expect("mouse selection must begin on the shown tab");
            assert_eq!(selection.terminal_id, terminal_id);

            // A later revision that still projects the terminal must not
            // clear the in-flight selection.
            let fresh = client.state_dump().unwrap();
            view.install_snapshot(fresh, cx);
            assert!(
                view.selection.is_some(),
                "selection must survive a snapshot reinstall"
            );

            // The full mousedown flow then applies the PaneFocus operation
            // result; selecting inside the already-active pane must not be
            // wiped by that step.
            let pane_id = view.focused_pane.unwrap();
            view.apply_operation_result(
                crate::command::OperationResult::PaneFocused { pane_id },
                cx,
            );
            assert!(
                view.selection.is_some(),
                "selection must survive the PaneFocus operation result"
            );
        });
        host.shutdown();
    }

    #[gpui::test]
    fn mouse_selection_targets_the_scrolled_content(cx: &mut gpui::TestAppContext) {
        // While the viewport is scrolled up (unacked fractional offset), a
        // pixel row must address the shifted source row, not the viewport's
        // first row.
        let mut host = crate::app::ModelHost::start();
        let client = std::sync::Arc::new(host.client());
        let workspace = client
            .dispatch(crate::command::AppCommand::Workspace(
                crate::command::WorkspaceCommand::Create,
            ))
            .unwrap();
        client.wait_operation(workspace).unwrap();
        let tab = client
            .dispatch(crate::command::AppCommand::Tab(
                crate::command::TabCommand::New { title: None },
            ))
            .unwrap();
        client.wait_operation(tab).unwrap();
        let dump = client.state_dump().unwrap();
        let terminal_id = displayed_terminal_id(&dump);
        client
            .dispatch(crate::command::AppCommand::Terminal(
                crate::command::TerminalCommand::SendText {
                    terminal_id: Some(terminal_id),
                    pane_id: None,
                    text: "for i in $(seq 1 40); do echo SCROLL_SEL_$i; done\n".to_owned(),
                },
            ))
            .unwrap();
        client
            .terminal_contains(
                terminal_id,
                "SCROLL_SEL_40",
                std::time::Duration::from_secs(5),
            )
            .unwrap();
        let mut snapshot = crate::terminal::snapshot_from_replay(
            &client.terminal_replay(terminal_id).unwrap(),
            2_000,
        );
        // The wire snapshot carries no grid; give the test snapshot a real
        // retained-history bound like the GUI local emulator would.
        snapshot.history_len = 40;
        snapshot.history_bottom = snapshot.viewport_position - 40;
        let snapshot = Arc::new(snapshot);

        let (view, cx) = cx.add_window_view(|_, cx| {
            WorkspaceView::new(client.clone(), dump.clone(), cx.focus_handle())
        });
        view.update_in(cx, |view, _window, cx| {
            view.terminal_snapshots.insert(terminal_id, snapshot);
            view.terminal_metrics = TerminalMetrics {
                cell_width: 8.0,
                line_height: 16.0,
                scale_factor: 1.0,
            };
            view.terminal_bounds
                .lock()
                .expect("bounds poisoned")
                .insert(
                    terminal_id,
                    Bounds::new(point(px(0.0), px(0.0)), size(px(640.0), px(384.0))),
                );
            // Simulate a 2.5-row unacked scroll up: the painted content is
            // shifted, so pixel row 1 addresses source row 3.
            view.scroll_accumulators
                .entry(terminal_id)
                .or_insert_with(|| TerminalScrollState::new(0))
                .visual_unacked_rows = 2.5;
            view.begin_terminal_selection(terminal_id, point(px(12.0), px(20.0)), false, cx);
            let selection = view
                .selection
                .expect("mouse selection must begin on the shown tab");
            assert_eq!(
                selection.anchor.position.row, -2,
                "a pixel in the shifted grid must map to the painted source row"
            );
            view.begin_terminal_selection(terminal_id, point(px(12.0), px(28.0)), false, cx);
            assert_eq!(
                view.selection.unwrap().anchor.position.row,
                -1,
                "the lower half after a fractional shift must map to the next painted row"
            );
            // With the viewport pinned at the bottom (no unacked offset), the
            // mapping is purely viewport-relative.
            view.scroll_accumulators
                .get_mut(&terminal_id)
                .unwrap()
                .visual_unacked_rows = 0.0;
            view.begin_terminal_selection(terminal_id, point(px(12.0), px(36.0)), false, cx);
            assert_eq!(
                view.selection.unwrap().anchor.position.row,
                2,
                "the mapping is viewport-relative without an unacked offset"
            );
            view.begin_terminal_selection(terminal_id, point(px(12.0), px(20.0)), false, cx);
            assert_eq!(
                view.selection.unwrap().anchor.position.row,
                1,
                "pixel row 1 maps to the viewport top row"
            );
            // Scrolled up again to a whole row: pixel row 1 addresses a
            // history row above the viewport (the mapping follows the
            // painted content, not the viewport top).
            view.scroll_accumulators
                .get_mut(&terminal_id)
                .unwrap()
                .visual_unacked_rows = 2.0;
            view.begin_terminal_selection(terminal_id, point(px(12.0), px(20.0)), false, cx);
            assert!(
                view.selection.unwrap().anchor.position.row < 0,
                "a whole-row unacked offset must shift the mapping into history"
            );
        });
        host.shutdown();
    }

    fn displayed_terminal_id(dump: &crate::app::model::ModelSnapshot) -> crate::ids::TerminalId {
        let workspace = dump.workspace.as_ref().expect("active workspace");
        let tab = &workspace.tabs[workspace.tabs.len() - 1];
        fn find(tree: &PaneTreeDump) -> Option<crate::ids::TerminalId> {
            match tree {
                PaneTreeDump::Leaf {
                    surface_state,
                    terminal,
                    ..
                } => match (surface_state, terminal.as_ref()) {
                    (SurfaceState::Terminal(state), Some(_)) => Some(state.terminal_id),
                    _ => None,
                },
                PaneTreeDump::Split { first, second, .. } => find(first).or_else(|| find(second)),
            }
        }
        find(&tab.tree).expect("displayed tab projects a grid")
    }

    fn assert_tree_is_terminal(tree: &PaneTreeDump) {
        match tree {
            PaneTreeDump::Leaf {
                surface_kind,
                surface_state,
                ..
            } => {
                assert_eq!(*surface_kind, crate::surface::SurfaceKind::Terminal);
                assert!(matches!(surface_state, SurfaceState::Terminal(_)));
            }
            PaneTreeDump::Split { first, second, .. } => {
                assert_tree_is_terminal(first);
                assert_tree_is_terminal(second);
            }
        }
    }

    /// Two independent model hosts whose first workspaces share the default
    /// title. Their UUID identities must remain distinct across connections.
    fn two_hosts_with_colliding_workspaces() -> (
        crate::app::ModelHost,
        crate::app::ModelHost,
        (WorkspaceId, String),
        (WorkspaceId, String),
    ) {
        let local_host = crate::app::ModelHost::start();
        let local_operation = local_host
            .client()
            .dispatch(AppCommand::Workspace(WorkspaceCommand::Create))
            .unwrap();
        local_host.client().wait_operation(local_operation).unwrap();
        let remote_host = crate::app::ModelHost::start();
        let remote_operation = remote_host
            .client()
            .dispatch(AppCommand::Workspace(WorkspaceCommand::Create))
            .unwrap();
        remote_host
            .client()
            .wait_operation(remote_operation)
            .unwrap();
        let local_dump = local_host.client().state_dump().unwrap();
        let remote_dump = remote_host.client().state_dump().unwrap();
        let (local_id, local_title) = {
            let workspace = local_dump
                .workspaces
                .first()
                .expect("local host creates a workspace");
            (workspace.id, workspace.title.clone())
        };
        let (remote_id, remote_title) = {
            let workspace = remote_dump
                .workspaces
                .first()
                .expect("remote host creates a workspace");
            (workspace.id, workspace.title.clone())
        };
        assert_ne!(
            local_id, remote_id,
            "independent hosts must use distinct UUIDs"
        );
        assert_eq!(
            local_title, remote_title,
            "test relies on a title collision"
        );
        (
            local_host,
            remote_host,
            (local_id, local_title),
            (remote_id, remote_title),
        )
    }

    #[gpui::test]
    fn rename_dialog_scopes_to_its_own_connection(cx: &mut gpui::TestAppContext) {
        let (mut local_host, mut remote_host, (_, _), (remote_ws, title)) =
            two_hosts_with_colliding_workspaces();
        let local_client: Arc<dyn CommandTransport> = Arc::new(local_host.client());
        let remote_client: Arc<dyn CommandTransport> = Arc::new(remote_host.client());
        let local_id = ConnectionId::new(1);
        let remote_id = ConnectionId::new(2);
        let connections = vec![
            WorkspaceConnection {
                id: local_id,
                title: "Local".to_owned(),
                kind: WorkspaceConnectionKind::Local,
                client: local_client,
                snapshot: local_host.client().state_dump().unwrap(),
            },
            WorkspaceConnection {
                id: remote_id,
                title: "build-box".to_owned(),
                kind: WorkspaceConnectionKind::Remote,
                client: remote_client,
                snapshot: remote_host.client().state_dump().unwrap(),
            },
        ];
        let (view, cx) = cx.add_window_view(move |_, cx| {
            WorkspaceView::new_with_connections(
                None,
                connections,
                local_id,
                cx.focus_handle(),
                AppConfig::default(),
            )
        });
        view.update_in(cx, |view, window, cx| {
            view.begin_rename_workspace(remote_id, remote_ws, window, cx);
        });
        view.update_in(cx, |view, _, _| {
            assert_eq!(
                view.dialog,
                Some(DialogState::RenameWorkspace {
                    connection_id: remote_id,
                    workspace_id: remote_ws
                })
            );
            assert_eq!(view.dialog_input, title);
            assert_eq!(view.dialog_caret, title.len());
            assert!(
                view.rename_target.is_none(),
                "workspace rename no longer uses the inline row caret"
            );
        });
        view.update_in(cx, |view, _, cx| view.confirm_dialog(cx));
        cx.run_until_parked();
        let remote_dump = remote_host.client().state_dump().unwrap();
        assert_eq!(
            remote_dump.workspaces[0].title, title,
            "confirming with the untouched name is a no-op rename"
        );

        // Rename the remote workspace and prove the colliding local row keeps
        // its own title.
        view.update_in(cx, |view, window, cx| {
            view.begin_rename_workspace(remote_id, remote_ws, window, cx);
        });
        view.update_in(cx, |view, _, cx| {
            view.dialog_input.clear();
            view.dialog_caret = 0;
            view.dialog_input.insert_str(0, "Build box");
            view.dialog_caret = 7;
            view.confirm_dialog(cx);
        });
        cx.run_until_parked();
        assert_eq!(
            remote_host.client().state_dump().unwrap().workspaces[0].title,
            "Build box"
        );
        assert_eq!(
            local_host.client().state_dump().unwrap().workspaces[0].title,
            title,
            "the colliding local workspace must stay untouched"
        );
        view.update_in(cx, |view, _, _| {
            assert_eq!(view.dialog, None);
            assert_eq!(
                view.connection_by_id(remote_id)
                    .unwrap()
                    .snapshot
                    .workspaces[0]
                    .title,
                "Build box"
            );
            assert_eq!(
                view.connection_by_id(local_id).unwrap().snapshot.workspaces[0].title,
                title
            );
        });
        local_host.shutdown();
        remote_host.shutdown();
    }

    #[gpui::test]
    fn text_input_dialog_confirm_is_inert_while_empty(cx: &mut gpui::TestAppContext) {
        let (mut local_host, mut remote_host, _, _) = two_hosts_with_colliding_workspaces();
        let local_client: Arc<dyn CommandTransport> = Arc::new(local_host.client());
        let remote_client: Arc<dyn CommandTransport> = Arc::new(remote_host.client());
        let local_id = ConnectionId::new(1);
        let remote_id = ConnectionId::new(2);
        let connections = vec![
            WorkspaceConnection {
                id: local_id,
                title: "Local".to_owned(),
                kind: WorkspaceConnectionKind::Local,
                client: local_client,
                snapshot: local_host.client().state_dump().unwrap(),
            },
            WorkspaceConnection {
                id: remote_id,
                title: "build-box".to_owned(),
                kind: WorkspaceConnectionKind::Remote,
                client: remote_client,
                snapshot: remote_host.client().state_dump().unwrap(),
            },
        ];
        let (view, cx) = cx.add_window_view(move |_, cx| {
            WorkspaceView::new_with_connections(
                None,
                connections,
                local_id,
                cx.focus_handle(),
                AppConfig::default(),
            )
        });
        // Empty connect-remote input: confirm must keep the dialog open.
        view.update_in(cx, |view, window, cx| view.begin_connect_remote(window, cx));
        view.update_in(cx, |view, _, cx| view.confirm_dialog(cx));
        view.update_in(cx, |view, _, _| {
            assert_eq!(view.dialog, Some(DialogState::ConnectRemote));
            assert!(!view.dialog_input_is_valid());
        });
        view.update_in(cx, |view, _, cx| view.cancel_dialog(cx));

        // Empty rename input: confirm must keep the dialog open and the model
        // untouched.
        view.update_in(cx, |view, window, cx| {
            view.begin_rename_workspace(remote_id, WorkspaceId::new(99), window, cx);
        });
        view.update_in(cx, |view, _, _| {
            assert!(
                view.dialog.is_none(),
                "renaming a workspace on a missing connection is a no-op"
            );
        });
        local_host.shutdown();
        remote_host.shutdown();
    }

    fn dialog_key_event(key: &str, key_char: Option<&str>) -> gpui::KeyDownEvent {
        gpui::KeyDownEvent {
            keystroke: gpui::Keystroke {
                modifiers: gpui::Modifiers::none(),
                key: key.to_owned(),
                key_char: key_char.map(str::to_owned),
            },
            is_held: false,
            prefer_character_input: false,
        }
    }

    #[gpui::test]
    fn dialog_input_caret_editing_follows_arrow_and_delete_keys(cx: &mut gpui::TestAppContext) {
        let (mut local_host, mut _remote_host, (local_ws, title), _) =
            two_hosts_with_colliding_workspaces();
        let local_client: Arc<dyn CommandTransport> = Arc::new(local_host.client());
        let local_id = ConnectionId::new(1);
        let (view, cx) = cx.add_window_view(|_, cx| {
            WorkspaceView::new(
                local_client.clone(),
                local_host.client().state_dump().unwrap(),
                cx.focus_handle(),
            )
        });
        view.update_in(cx, |view, window, cx| {
            view.begin_rename_workspace(local_id, local_ws, window, cx);
        });
        // Prefilled title, caret at the end: "Workspace 1|".
        view.update_in(cx, |view, _, cx| {
            view.handle_dialog_key(&dialog_key_event("left", None), cx)
        });
        view.update_in(cx, |view, _, _| {
            assert_eq!(view.dialog_caret, title.len() - 1);
            assert_eq!(
                &view.dialog_input[..view.dialog_caret],
                &title[..title.len() - 1]
            );
            assert_eq!(&view.dialog_input[view.dialog_caret..], "1");
        });
        // Delete removes the character after the caret ("1").
        view.update_in(cx, |view, _, cx| {
            view.handle_dialog_key(&dialog_key_event("delete", None), cx)
        });
        view.update_in(cx, |view, _, _| {
            assert_eq!(view.dialog_input, "Workspace ");
            assert_eq!(view.dialog_caret, "Workspace ".len());
        });
        // Home, then type at the start.
        view.update_in(cx, |view, _, cx| {
            view.handle_dialog_key(&dialog_key_event("home", None), cx)
        });
        view.update_in(cx, |view, _, cx| {
            view.handle_dialog_key(&dialog_key_event("x", Some("x")), cx)
        });
        view.update_in(cx, |view, _, _| {
            assert_eq!(view.dialog_input, "xWorkspace ");
            assert_eq!(view.dialog_caret, 1);
        });
        view.update_in(cx, |view, _, cx| view.cancel_dialog(cx));
        local_host.shutdown();
    }

    #[gpui::test]
    fn dialog_scrim_covers_the_whole_window_centered(cx: &mut gpui::TestAppContext) {
        let mut host = crate::app::ModelHost::start();
        let client = std::sync::Arc::new(host.client());
        let operation = client
            .dispatch(AppCommand::Workspace(WorkspaceCommand::Create))
            .unwrap();
        client.wait_operation(operation).unwrap();
        let snapshot = client.state_dump().unwrap();
        let (view, cx) = cx.add_window_view(|_, cx| {
            WorkspaceView::new(client.clone(), snapshot.clone(), cx.focus_handle())
        });
        view.update_in(cx, |view, _window, cx| {
            view.dialog = Some(DialogState::ConnectRemote);
            cx.notify();
        });
        cx.run_until_parked();
        let window_size = view.update_in(cx, |_, window, _| window.bounds().size);
        let scrim = cx
            .debug_bounds("dialog-scrim")
            .expect("dialog scrim is rendered");
        assert_eq!(scrim.size, window_size, "the scrim must cover the window");
        let center = gpui::point(
            px(f32::from(scrim.origin.x) + f32::from(scrim.size.width) / 2.0),
            px(f32::from(scrim.origin.y) + f32::from(scrim.size.height) / 2.0),
        );
        let window_center = gpui::point(
            px(f32::from(window_size.width) / 2.0),
            px(f32::from(window_size.height) / 2.0),
        );
        assert_eq!(center, window_center, "the dialog must be window-centered");
        host.shutdown();
    }
}
