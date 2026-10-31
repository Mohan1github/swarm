from __future__ import annotations

import queue
import threading
import tkinter as tk
from tkinter import ttk

from agents.orchestrator import ReasoningSwarmEngine
from graph_visualizer import build_live_execution_report
from llm_client import build_llm_adapter


NODE_POSITIONS = {
    "planner": (150, 150),
    "researcher": (330, 100),
    "analyst": (520, 150),
    "critic": (700, 220),
    "executor": (520, 380),
    "validator": (330, 430),
    "synthesizer": (150, 380),
}


class SwarmVisualizerApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Swarm Intelligence Visualizer")
        self.root.geometry("1280x860")
        self.root.minsize(1100, 720)

        self.root.configure(bg="#0b1020")
        self.style = ttk.Style(self.root)
        self.style.theme_use("clam")
        self.style.configure("TFrame", background="#0b1020")
        self.style.configure("TLabel", background="#0b1020", foreground="#eaf2ff")
        self.style.configure("TButton", padding=(16, 10), relief="flat", background="#1a2a4a", foreground="#eef6ff")
        self.style.map("TButton", background=[("active", "#243f68")])
        self.style.configure("TEntry", fieldbackground="#111b2d", foreground="#f0f6ff", borderwidth=0)
        self.style.configure("Treeview", background="#101827", foreground="#edf6ff", fieldbackground="#101827")
        self.style.configure("Treeview.Heading", background="#1e2d4d", foreground="#edf6ff", relief="flat")
        self.style.map("Treeview", background=[("selected", "#334d8a")])

        self.query_var = tk.StringVar(value="How do autonomous agents coordinate under load?")

        background = tk.Frame(root, bg="#0b1020")
        background.pack(fill="both", expand=True)

        header = tk.Frame(background, bg="#101a2d", bd=0, highlightthickness=0)
        header.pack(fill="x", padx=12, pady=(12, 0))
        tk.Label(header, text="SWARM INTELLIGENCE VISUALIZER", bg="#101a2d", fg="#eaf2ff", font=("Segoe UI", 16, "bold")).pack(anchor="w", padx=14, pady=(12, 10))

        top = tk.Frame(background, bg="#101a2d", padx=14, pady=12)
        top.pack(fill="x", padx=12, pady=12)

        tk.Label(top, text="User Query", bg="#101a2d", fg="#dcecfb", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        query_box = tk.Entry(
            top,
            textvariable=self.query_var,
            bg="#0d1526",
            fg="#f1f5ff",
            insertbackground="#f1f5ff",
            bd=0,
            relief="flat",
            font=("Segoe UI", 12),
        )
        query_box.pack(fill="x", pady=(8, 12), ipady=9)

        actions = tk.Frame(top, bg="#101a2d")
        actions.pack(fill="x")
        run_btn = tk.Button(actions, text="Run Swarm", bg="#3b82f6", fg="#ffffff", bd=0, relief="flat", padx=20, pady=10, font=("Segoe UI", 10, "bold"), command=self.run_query)
        run_btn.pack(side="left", padx=(0, 10))
        self.run_button = run_btn
        reset_btn = tk.Button(actions, text="Reset", bg="#1d2d4a", fg="#e5ecff", bd=0, relief="flat", padx=18, pady=10, font=("Segoe UI", 10, "bold"), command=self.reset_view)
        reset_btn.pack(side="left")

        main = tk.PanedWindow(background, orient="horizontal", bg="#0b1020")
        main.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        graph_panel = tk.Frame(main, bg="#0f172a", bd=0, highlightthickness=0)
        graph_panel.pack(fill="both", expand=True)
        graph_panel.configure(padx=10, pady=10)
        self.canvas = tk.Canvas(graph_panel, width=760, height=520, bg="#0b1324", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        stream_frame = tk.Frame(graph_panel, bg="#0f172a")
        stream_frame.pack(fill="x", padx=10, pady=(0, 10))
        tk.Label(stream_frame, text="Live Execution Stream", bg="#0f172a", fg="#ecf5ff", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.stream = tk.Text(stream_frame, height=7, wrap="word", bg="#0d1526", fg="#edf6ff", insertbackground="#edf6ff", font=("Segoe UI", 9))
        self.stream.pack(fill="x", expand=True)

        self.timeline = tk.Frame(stream_frame, bg="#0f172a")
        self.timeline.pack(fill="x", pady=(6, 0))
        self.timeline_steps = []

        side = tk.Frame(main, bg="#111b2d", width=320)
        side.pack(fill="y")

        self.metrics_frame = tk.Frame(side, bg="#111b2d")
        self.metrics_frame.pack(fill="x", padx=12, pady=(8, 10))
        self.llm_adapter = build_llm_adapter()
        self.workflow_events = queue.Queue()
        self.workflow_running = False
        self.selected_agent_count = 0
        self.current_action_probabilities = {}
        self.completed_stage_roles = []
        self.validation_score = None

        self.metric_cards = {}
        for title, value in (("STATUS", "READY"), ("LATENCY", "--"), ("AGENTS", "--"), ("VALIDATION", "--"), ("STAGES", "0/7"), ("MODEL", "--")):
            card = tk.Frame(self.metrics_frame, bg="#0d1526", padx=10, pady=8)
            card.pack(fill="x", pady=4)
            tk.Label(card, text=title, bg="#0d1526", fg="#9db9d9", font=("Segoe UI", 9, "bold")).pack(anchor="w")
            label = tk.Label(card, text=value, bg="#0d1526", fg="#edf6ff", font=("Segoe UI", 11, "bold"))
            label.pack(anchor="w")
            self.metric_cards[title.lower()] = label

        self.service_frame = tk.Frame(side, bg="#111b2d")
        self.service_frame.pack(fill="x", padx=12, pady=(0, 10))
        self.service_cards = {}
        for title, value in (("SERVICE", "READY"), ("STAGES", "0/7"), ("LATENCY", "--"), ("MODE", "LLM LIVE")):
            card = tk.Frame(self.service_frame, bg="#0d1526", padx=10, pady=8)
            card.pack(fill="x", pady=4)
            tk.Label(card, text=title, bg="#0d1526", fg="#9db9d9", font=("Segoe UI", 9, "bold")).pack(anchor="w")
            label = tk.Label(card, text=value, bg="#0d1526", fg="#edf6ff", font=("Segoe UI", 11, "bold"))
            label.pack(anchor="w")
            self.service_cards[title.lower()] = label

        tk.Label(side, text="Probabilistic Action Plan", bg="#111b2d", fg="#ecf5ff", font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=12, pady=(12, 6))
        self.tree = ttk.Treeview(side, columns=("action", "probability"), show="headings", height=16)
        self.tree.heading("action", text="Action")
        self.tree.heading("probability", text="Probability")
        self.tree.column("action", width=165)
        self.tree.column("probability", width=110, anchor="center")
        self.tree.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        tk.Label(side, text="Final Synthesis", bg="#111b2d", fg="#ecf5ff", font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=12, pady=(0, 6))
        self.summary = tk.Text(side, height=15, wrap="word", bg="#0d1526", fg="#edf6ff", insertbackground="#edf6ff", font=("Segoe UI", 10))
        self.summary.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        self.node_ids = {}
        self.reset_view()

    def reset_view(self) -> None:
        self.canvas.delete("all")
        self.canvas.create_rectangle(0, 0, 760, 620, fill="#0b1324", outline="")
        self.canvas.create_text(380, 290, text="Run the swarm to visualize execution", fill="#dfeaf3", font=("Segoe UI", 16, "bold"))
        self.canvas.create_text(380, 325, text="Probability-weighted reasoning pipeline", fill="#88a8d8", font=("Segoe UI", 10))
        self.tree.delete(*self.tree.get_children())
        self.summary.delete("1.0", "end")
        self.summary.insert("end", "Configure a Groq API key and run a query to execute the live model workflow.")
        self.stream.delete("1.0", "end")

        for key in ("status", "latency", "agents", "validation", "stages", "model"):
            self.metric_cards[key].config(text="--")
        for key in ("service", "stages", "latency", "mode"):
            self.service_cards[key].config(text="--")

    def _draw_node(self, cx: int, cy: int, label: str, probability: float, color: str, role: str = "") -> object:
        radius = 52
        fill = color if probability > 0.0 else "#3f4f5f"
        node_id = self.canvas.create_oval(cx - radius, cy - radius, cx + radius, cy + radius, fill=fill, outline="#dfeaf3", width=2, tags=("node", role))
        text_id = self.canvas.create_text(cx, cy - 4, text=label, fill="#ffffff", font=("Segoe UI", 10, "bold"), justify="center", tags=("node_text", role))
        prob_id = self.canvas.create_text(cx, cy + 20, text=f"p={probability:.3f}", fill="#dfeaf3", font=("Segoe UI", 8), tags=("node_prob", role))
        return {"oval": node_id, "text": text_id, "prob_text": prob_id}

    def _draw_edge(self, a: tuple[int, int], b: tuple[int, int]) -> None:
        self.canvas.create_line(a[0], a[1], b[0], b[1], fill="#7fa9ff", width=2, arrow=tk.LAST)

    def _draw_graph(self, report: dict) -> None:
        self.canvas.delete("all")
        pipeline_by_role = {item["role"]: item for item in report["pipeline"]}

        colors = {
            "planner": "#5fa8ff",
            "researcher": "#6ee7b7",
            "analyst": "#fbbf24",
            "critic": "#f87171",
            "executor": "#c084fc",
            "validator": "#34d399",
            "synthesizer": "#f472b6",
        }

        connections = [
            ("planner", "researcher"),
            ("researcher", "analyst"),
            ("analyst", "critic"),
            ("critic", "executor"),
            ("executor", "validator"),
            ("validator", "synthesizer"),
            ("executor", "synthesizer"),
        ]

        for a, b in connections:
            self._draw_edge(NODE_POSITIONS[a], NODE_POSITIONS[b])

        self.node_ids = {}
        for role in NODE_POSITIONS:
            node = pipeline_by_role.get(role)
            if node is None:
                continue
            cx, cy = NODE_POSITIONS[role]
            node_map = self._draw_node(cx, cy, role.upper(), node["probability"], colors.get(role, "#8bb5ff"), role)
            self.node_ids[role] = node_map
            if node["status"] == "completed":
                self.canvas.itemconfig(node_map["oval"], fill="#2f6fee", outline="#d9f99d", width=4)

        self.canvas.create_text(380, 28, text="SWARM EXECUTION GRAPH", fill="#edf6ff", font=("Segoe UI", 16, "bold"))
        validation_score = report["validation_score"]
        validation_label = "pending" if validation_score is None else f"{validation_score:.2f}"
        self.canvas.create_text(380, 590, text=f"Workflow: sequential LLM | Validation self-assessment: {validation_label}", fill="#d5e7ff", font=("Segoe UI", 10))
        self.canvas.create_rectangle(30, 46, 730, 560, outline="#213555", width=1)

    def _update_live_metrics(self, workflow: dict) -> None:
        stage_count = len(workflow["steps"])
        self.metric_cards["status"].config(text="COMPLETE")
        self.metric_cards["latency"].config(text=f"{workflow['latency_ms']:.0f} ms")
        self.metric_cards["agents"].config(text=str(self.selected_agent_count))
        self.metric_cards["validation"].config(text=f"{workflow['validation_score']:.2f}")
        self.metric_cards["stages"].config(text=f"{stage_count}/7")
        self.metric_cards["model"].config(text=workflow["model"])
        self.service_cards["service"].config(text="LLM LIVE")
        self.service_cards["stages"].config(text=f"{stage_count}/7")
        self.service_cards["latency"].config(text=f"{workflow['latency_ms']:.0f} ms")
        self.service_cards["mode"].config(text="SEQUENTIAL")

    def _render_timeline(self, report: dict) -> None:
        for widget in self.timeline.winfo_children():
            widget.destroy()
        self.timeline_steps = []
        for node in report["pipeline"]:
            chip = tk.Label(
                self.timeline,
                text=node["role"].upper(),
                bg="#101827",
                fg="#dfeaf3",
                padx=8,
                pady=4,
                font=("Segoe UI", 8, "bold"),
                bd=0,
                relief="flat",
            )
            chip.pack(side="left", padx=3)
            if node["status"] == "completed":
                chip.config(bg="#17314d", fg="#e8f0ff")
            self.timeline_steps.append(chip)

    def run_query(self) -> None:
        query = self.query_var.get().strip()
        if not query or self.workflow_running:
            return

        if not self.llm_adapter.enabled:
            self.service_cards["service"].config(text="CONFIG REQUIRED")
            self.summary.delete("1.0", "end")
            self.summary.insert("end", "Live LLM is not configured. Set GROQ_API_KEY and restart the application.")
            return

        self.workflow_running = True
        self.run_button.config(state="disabled")
        self.service_cards["service"].config(text="RUNNING")
        self.metric_cards["status"].config(text="RUNNING")
        self.current_action_probabilities = {}
        self.completed_stage_roles = []
        self.validation_score = None
        self.stream.delete("1.0", "end")
        self.summary.delete("1.0", "end")
        self.summary.insert("end", "Waiting for the live workflow to complete...")

        def execute() -> None:
            try:
                engine = ReasoningSwarmEngine(total_agents=10_000)
                for event in engine.iter_workflow(query):
                    self.workflow_events.put(event)
                    if event["type"] == "error":
                        return
            except Exception:
                self.workflow_events.put({
                    "type": "error",
                    "stage": "workflow",
                    "message": "The live workflow failed. Check provider configuration and server logs.",
                })

        threading.Thread(target=execute, daemon=True).start()
        self.root.after(100, self._poll_workflow_events)

    def _poll_workflow_events(self) -> None:
        try:
            while True:
                event = self.workflow_events.get_nowait()
                event_type = event["type"]
                if event_type == "setup":
                    self.selected_agent_count = event["selected_agents"]
                    self.metric_cards["agents"].config(text=str(self.selected_agent_count))
                elif event_type == "step":
                    self._display_live_step(event)
                elif event_type == "complete":
                    self._finish_live_workflow(event)
                elif event_type == "error":
                    self._fail_live_workflow(event)
        except queue.Empty:
            pass

        if self.workflow_running:
            self.root.after(100, self._poll_workflow_events)

    def _display_live_step(self, step: dict) -> None:
        role = step["role"]
        self.stream.insert("end", f"[{role.upper()}] completed in {step['duration_ms']:.0f} ms\n{step['output']}\n\n")
        self.stream.see("end")

        if role == "planner":
            self.current_action_probabilities = step["action_probabilities"]
            self.tree.delete(*self.tree.get_children())
            for action, probability in sorted(step["action_probabilities"].items(), key=lambda item: item[1], reverse=True):
                self.tree.insert("", "end", values=(action, f"{probability:.4f}"))

        if "validation_score" in step:
            self.validation_score = step["validation_score"]
        if role not in self.completed_stage_roles:
            self.completed_stage_roles.append(role)

        if self.current_action_probabilities:
            report = build_live_execution_report(
                self.current_action_probabilities,
                self.completed_stage_roles,
                self.validation_score,
            )
            self._draw_graph(report)
            if role == "planner":
                self._render_timeline(report)
            for chip in self.timeline_steps:
                if chip.cget("text").lower() in self.completed_stage_roles:
                    chip.config(bg="#17314d", fg="#e8f0ff")

    def _finish_live_workflow(self, workflow: dict) -> None:
        self.current_action_probabilities = workflow["action_probabilities"]
        self.completed_stage_roles = [step["role"] for step in workflow["steps"]]
        self.validation_score = workflow["validation_score"]
        report = build_live_execution_report(
            workflow["action_probabilities"],
            self.completed_stage_roles,
            self.validation_score,
        )
        self._draw_graph(report)
        self._update_live_metrics(workflow)
        for chip in self.timeline_steps:
            chip.config(bg="#17314d", fg="#e8f0ff")
        self.summary.delete("1.0", "end")
        self.summary.insert("end", workflow["answer"])
        self._end_live_workflow()

    def _fail_live_workflow(self, event: dict) -> None:
        self.metric_cards["status"].config(text="FAILED")
        self.service_cards["service"].config(text="LLM ERROR")
        self.stream.insert("end", f"[ERROR] {event['stage']}: {event['message']}\n")
        self.stream.see("end")
        self.summary.delete("1.0", "end")
        self.summary.insert("end", f"Live execution stopped at {event['stage']}. No simulated answer was substituted.")
        self._end_live_workflow()

    def _end_live_workflow(self) -> None:
        self.workflow_running = False
        self.run_button.config(state="normal")


if __name__ == "__main__":
    root = tk.Tk()
    app = SwarmVisualizerApp(root)
    root.mainloop()
