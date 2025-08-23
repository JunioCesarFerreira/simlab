import matplotlib.pyplot as plt
from pathlib import Path

import data_analysis

# Processa e plota dados de logs com várias métricas de teste
def process_log1(log_path: Path, csv_full_output: Path, csv_means_output: Path):
    # -------------------------- DataFrame bruto --------------------------------
    df = data_analysis.convert_log_to_csv(log_path, csv_full_output)

    # Quantidades de pacotes perdidos
    df["lost_packets_r2n"] = df["server_sent"] - df["total_received"] 
    df["lost_packets_n2r"] = df["server_sent"] + df["total_sent"] - df["server_received"]

    # ------------------------- Médias por mote ---------------------------------
    metrics_cols = [
        "rtt_latency",
        "r2n_latency",
        "n2r_latency",
        "rssi",
        "radio_rx_energy_mj",
        "radio_tx_energy_mj",
        "cpu_energy_mj",
        "lost_packets_r2n",
        "lost_packets_n2r",
        "hops",
    ]

    means = (
        df.groupby("node")[metrics_cols]
        .mean()
        .round(2)            # duas casas decimais para facilitar leitura
        .reset_index()
    )

    means.to_csv(csv_means_output, index=False)
    print("\n=== MÉDIAS POR MOTE ===")
    print(means.to_string(index=False))
    # --------------------------- Plots -----------------------------------------
    unique_nodes = df["node"].unique()

    for metric in metrics_cols:
        plt.figure(figsize=(12, 6))
        for node in unique_nodes:
            node_df = df[df["node"] == node]
            plt.plot(
                node_df["root_time_now"],
                node_df[metric],
                label=f"{node}",
            )
        plt.title(f"{metric} ao longo do tempo")
        plt.xlabel("root_time_now (ms)")
        plt.ylabel(metric)
        plt.legend(loc="upper left", fontsize="small")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.show()
        

# Processa e plota logs de 6 métricas (usada em trabalho precedente)
def process_log2(log_path: Path, csv_full_output: Path, csv_means_output: Path):
    # -------------------------- DataFrame bruto --------------------------------
    df = data_analysis.convert_log_to_csv(log_path, csv_full_output)

    # ------------------------- Médias por mote ---------------------------------
    metrics_cols = [
        "cpu_mj",
        "total_mj",
        "latency_ms",
        "volume_B",
        "response_ms",
        "throughput_bps",
    ]

    means = (
        df.groupby("node")[metrics_cols]
        .mean()
        .round(2)            # duas casas decimais para facilitar leitura
        .reset_index()
    )

    means.to_csv(csv_means_output, index=False)
    print("\n=== MÉDIAS POR MOTE ===")
    print(means.to_string(index=False))
    # --------------------------- Plots -----------------------------------------
    unique_nodes = df["node"].unique()

    for metric in metrics_cols:
        plt.figure(figsize=(12, 6))
        for node in unique_nodes:
            node_df = df[df["node"] == node]
            plt.plot(
                node_df["root_time_now"],
                node_df[metric],
                label=f"{node}",
            )
        plt.title(f"{metric} ao longo do tempo")
        plt.xlabel("root_time_now (ms)")
        plt.ylabel(metric)
        plt.legend(loc="upper left", fontsize="small")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.show()