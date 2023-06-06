import matplotlib.pyplot as plt


labels = [
    "Junior Software Engineer",
    "Senior Software Engineer",
    "Software Engineer",
    "System Architect",
    "Technical Lead",
]

data = [63, 31, 100, 2, 11]
explode = [0.15, 0, 0, 0, 0]
plt.pie(
    data,
    labels=labels,
    shadow=True,
    explode=explode,
    autopct="%.2f%%",
    pctdistance=1.15,
    labeldistance=1.35,
)

plt.show()


# --- pie ----
