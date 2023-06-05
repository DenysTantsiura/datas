import pandas as pd

mountains_height = pd.Series(
    data=[2061, 2035.8, 2028.5, 2022.5, 2016.4],
    index=["Goverla", "Brebenskyl", "Pip_Ivan", "Petros", "Gutin_Tomnatik"],
    name="Height, m",
    dtype=float,
)

print(mountains_height[1:3])
print(mountains_height["Brebenskyl":"Petros"])
print(mountains_height.Petros)  # 2022.5
print(mountains_height.Brebenskyl)  # 2035.8

print(mountains_height > 2030)
print(mountains_height[mountains_height > 2030])

print("Goverla" in mountains_height)  # True

sort_index = mountains_height.sort_index()
print(sort_index)

mountains_height.sort_values(inplace=True, ascending=False)
print(mountains_height)

