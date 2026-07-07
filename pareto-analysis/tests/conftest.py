"""Force a non-interactive matplotlib backend before plot_pareto_results
(which imports matplotlib.pyplot at module load) is collected."""
import matplotlib

matplotlib.use("Agg")
