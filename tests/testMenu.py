import sys, os
from TUI.IideaTUI import IideaAnalyzer, IideaMenu, console

options = [
    "conv1 : Conv2d(3, 64, kernel_size=(7, 7))",
    "bn1 : BatchNorm2d(64)",
    "relu : ReLU(inplace=True)",
    "layer1 : Sequential",
    "layer2 : Sequential",
    "avgpool : AdaptiveAvgPool2d(output_size=(1, 1))",
    "fc : Linear(in_features=512, out_features=1000)",
]

menu = IideaMenu(options)
choice = menu.select()

console.print("")

if choice == -1:
    console.print("backed out (-1)", style="dark_orange")
else:
    console.print(f"selected {choice}) {options[choice]}", style="dark_orange")
