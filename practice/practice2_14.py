import matplotlib.pyplot as plt


plt.bar(
    ["США", "Китай", "Японія", "Велика Британія"],
    [39, 38, 27, 22],
    color=["b", "r", "y", "g"],
)

plt.xlabel("Країна", fontsize="small", color="midnightblue")
plt.ylabel("Кількість", fontsize="small", color="midnightblue")
plt.title("Золоті медалі: Літня олімпіада Токіо 2020", fontsize=15)
plt.show()

# --- gist color ----
