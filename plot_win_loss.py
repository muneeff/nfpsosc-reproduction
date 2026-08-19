import matplotlib.pyplot as plt


labels = [
    "PC_NFPSO",
    "NF_BASE",
    "Tie"
]

values = [
    3720,
    4409,
    651
]


plt.figure(
    figsize=(6,4)
)

plt.bar(
    labels,
    values
)

plt.ylabel(
    "Number of Tasks"
)

plt.title(
    "PC_NFPSO vs NF_BASE Win/Loss"
)

plt.tight_layout()

plt.savefig(
    "paper_tables/win_loss.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print("Saved win_loss.png")