import os
import time
import sys
import torch
import readchar
import pyfiglet
import numpy as np
import matplotlib.pyplot as plt
from rich.console import Console, Group
from rich.live import Live
from rich.text import Text
from rich.panel import Panel
from rich.rule import Rule
from rich.prompt import Prompt

console         = Console()
orange_line     = Rule(style="dark_orange")
default_color   = "dark_orange"

# --- key bindings for menus, names indicate the action ---
menu_controls   = {"up": readchar.key.UP,
                   "down": readchar.key.DOWN,
                   "select": [readchar.key.ENTER, readchar.key.RIGHT],
                   "back": readchar.key.LEFT,
                   "exit": readchar.key.CTRL_E}

# --- key bindings for the error box, names indicate the action ---
error_box_controls  = {"back": [readchar.key.LEFT, readchar.key.ENTER],
                       "exit": readchar.key.CTRL_E}

# block_format {"name": ..., "type": ..., "id": ..., "extra_repr": ...}

class IideaAnalyzer:
    def __init__(self,
                 model_ptr: torch.nn.Module) -> None:

        self.model              = model_ptr
        self.named_params       = self.model.named_parameters()
        self.current_pointer    = self.model
        self.previous_pointers  = []

    @staticmethod
    def print_controls() -> None:
        content = Text("back: ← | up ↑ | down ↓ | select: enter/→ | exit: CTRL E", style="white")
        console.print(Panel(content, border_style="dark_orange", expand=False))

    @staticmethod
    def print_logo() -> None:
        logolines = pyfiglet.figlet_format("Iidea MI", font="big").rstrip()
        logo = Text(logolines, style="dark_orange")
        sublogo = Text("  Pytorch Deep Model Analysis ", style="white")

        console.clear()
        console.print("\n")
        console.print(orange_line)
        console.print(logo)
        console.print(orange_line)
        console.print(sublogo)
        console.print(orange_line)

    def main(self) -> None:
        IideaAnalyzer.print_logo()

        while True:
            menu, formatted_children = IideaAnalyzer.render_node_menu(self.current_pointer)
            menu_selection = menu.select()
            # --- main menu ---
            if isinstance(menu_selection, int):
                self.previous_pointers.append(self.current_pointer)
                self.current_pointer = getattr(self.current_pointer, formatted_children[menu_selection]["name"])
                continue

            # --- special exits ---
            elif isinstance(menu_selection, str):
                # --- back special exit ---
                if menu_selection == "back":
                    if len(self.previous_pointers) == 0: continue
                    self._back_update_pointers()
                    continue

                # --- empty special exit (assume it is a leaf) ---
                if menu_selection == "empty":

                    # --- leaf menu (weight: ..., bias ...) ---
                    leaf_menu, leaf_formatted_options = IideaAnalyzer.render_leaf_menu(self.current_pointer)
                    leaf_menu_selection = leaf_menu.select()

                    if isinstance(leaf_menu_selection, int):
                        # --- display plot menu and do not update pointer ---
                        opt = leaf_formatted_options[leaf_menu_selection]
                        self.render_plot_subui(opt["param"], opt["param_type"], opt["param_shape"], f"{self.current_pointer}")

                     # --- special exits if leaf contains no parameters ---
                    elif isinstance(leaf_menu_selection, str):
                        # --- back special exit ---
                        if leaf_menu_selection == "back":
                            self._back_update_pointers()
                            continue

                        # --- if leaf is empty display error box ---
                        if leaf_menu_selection == "empty":
                            choices = error_box_controls["back"]
                            key = IideaAnalyzer.render_error_box(f"{self.current_pointer} did not return any parameters!", allowed_key_returns=choices)
                            if key in choices:
                                self._back_update_pointers()

            else:
                raise TypeError(f"Could not resolve type {type(menu_selection).__name__}:{menu_selection}")

    def _back_update_pointers(self) -> None:
        self.current_pointer = self.previous_pointers[-1]
        self.previous_pointers.pop(-1)

    @staticmethod
    def get_formatted_children(module: torch.nn.Module) -> list[dict]:
        formatted_children = []
        for name, child in module.named_children():

            formatted_children.append({"name": name,
                                       "type": type(child).__name__ ,
                                       "id": hash(child),
                                       "extra_repr": child.extra_repr()})

        return formatted_children

    @staticmethod
    def get_formatted_parameters(module: torch.nn.Module) -> list[dict]:
        formatted_children = []
        for name, param in module.named_parameters():

            formatted_children.append({"name": name,
                                       "param_type": type(param).__name__,
                                       "param": param,
                                       "param_shape": param.shape})
        return formatted_children

    @staticmethod
    def render_error_box(message: str, allowed_key_returns: list[str] = error_box_controls["back"]) -> str:
        content = Text(f"| {message} | back: ←/ENTER | exit: CTRL E |")
        panel = Panel(content, border_style="red", expand=False)

        with Live(panel, console=console, auto_refresh=False, transient=True) as live:
            live.refresh()

            while True:
                key = readchar.readkey()

                if key == error_box_controls["exit"]:
                    sys.exit()

                elif key in allowed_key_returns:
                    return key

    @staticmethod
    def render_node_menu(current_pointer: torch.nn.Module) -> tuple["IideaMenu", list[dict]]:
        formatted_options = IideaAnalyzer.get_formatted_children(current_pointer)
        options = [item["name"] for item in formatted_options]
        menu = IideaMenu(options)
        return menu, formatted_options

    @staticmethod
    def render_leaf_menu(current_pointer: torch.nn.Module) -> tuple["IideaMenu", list[dict]]:
        formatted_options = IideaAnalyzer.get_formatted_parameters(current_pointer)
        options = [f"{item["name"]} | Shape: {item["param_shape"]}" for item in formatted_options]
        menu = IideaMenu(options)
        return menu, formatted_options

    def render_plot_subui(self,
                          parameter: torch.Tensor,
                          parameter_type: str,
                          parameter_shape: torch.Size,
                          parent_module_name: str) -> None:

        content = Text(f"Selected: {parameter_type} | "
                       + f"Name: {parent_module_name} | "
                       + f"Shape: {parameter_shape}" )

        panel = Panel(content, border_style="dark_orange", expand=False)

        tensor_plot_menu = IideaMenu(["Plot 1-D", "Plot 2-D"], panel=panel)

        plot_type_choice = tensor_plot_menu.select()

        if isinstance(plot_type_choice, str):
            if plot_type_choice == "back":
                # no need to do a _back_update_pointers because ptrs where never updated originally to the plot menu
                # a simple return moves you back to the previous node 
                return

        if isinstance(plot_type_choice, int):
            console.print(panel)

            while True:
                tensor_shape_str = Prompt.ask("Please Index the tensor you want to plot:", default="[1, 64, :]")
                tensor_slices = IideaAnalyzer.validate_shape(tensor_shape_str, parameter_shape)

                if tensor_slices is None:
                    console.print(Text("[Iidea] User Warning: Failed to index!", style = "red"))
                else:
                    plottable = IideaAnalyzer.convert_tensor(parameter[*tensor_slices])
                    plottable = plottable.squeeze()
                    print(plottable.size)

                    if plot_type_choice == 0:
                        if plottable.ndim != 1:
                            console.print(Text("[Iidea] User Warning: (plottable.ndim != 1), ensure indexing is ':n' vs 'n'!", style = "red"))
                            continue
                        plot1d(plottable)

                    elif plot_type_choice == 1:
                        print(plottable.ndim)
                        if plottable.ndim != 2:
                            console.print(Text("[Iidea] User Warning: (plottable.ndim != 2), ensure indexing is ':n' vs 'n'!", style = "red"))
                            continue
                        plot2d(plottable)

                    console.print(Text("[Iidea] Sucessfully plotted!", style = "green"))
                    time.sleep(2)
                    # --- soft reset ui ---
                    self.print_logo()
                    return
                    
    @staticmethod
    def validate_shape(tensor_shape_str: str, actual_shape: tuple, debug = False) -> tuple | None:
        try:
            ts_charlist = list(tensor_shape_str)

            # --- clean and preprocess str into list ---
            ndims           = 0
            ts_cleaned_charlist  = []

            for i, char in enumerate(ts_charlist):
                if char == ",":
                    ndims += 1
                    ts_cleaned_charlist.append(char)
                if (char == ":") or char.isnumeric():
                    ts_cleaned_charlist.append(char)
                else:
                    pass

            if debug: print("cleaned:", ts_cleaned_charlist)

            assert(len(actual_shape) == ndims + 1)
            
            new_tensor_shape_str = ""
            for c in ts_cleaned_charlist:
                new_tensor_shape_str = new_tensor_shape_str + c

            if debug: print("cleaned str:", new_tensor_shape_str)

            tensor_list = new_tensor_shape_str.split(",")

            if debug: print("cleaned split:", tensor_list)

            slices = []
            for t in tensor_list:
                default = [None, None, None]
                for i, c in enumerate(t.split(":")):
                    if c != "":
                        default[i] = int(c)

                if debug: print("slice", default)
                slices.append(slice(*default))

            if debug: print("slices", slices)
            return slices
        except:
            return None

    @staticmethod
    def convert_tensor(tensor: torch.Tensor) -> np.ndarray:
        return tensor.detach().cpu().numpy().astype(float)

class IideaMenu:
    def __init__(self,
                 options: list,
                 number_style: str = "dark_orange",
                 text_style: str = "white",
                 highlight_style: str = "dark_orange",
                 panel: Panel | None = None) -> None:

        self.options            = list(options)
        self.number_style       = number_style
        self.text_style         = text_style
        self.highlight_style    = highlight_style
        self.index              = 0
        self.panel              = panel

    def render(self) -> Text | Group:
        menu = Text()
        for i, option in enumerate(self.options):
            menu.append(f" {i}) ", style=self.number_style)

            if i == self.index:
                menu.append(f"{option}\n", style=self.highlight_style + " bold")
            else:
                menu.append(f"{option}\n", style=self.text_style + " bold")

        if self.panel is not None:
            return Group(self.panel, menu)
        else:
            return menu

    def update(self, step: int) -> None:
        moved = self.index + step
        if 0 <= moved < len(self.options):
            self.index = moved

    def select(self) -> int | str:

        if len(self.options) == 0:
            return self.empty()

        with Live(self.render(), console=console, auto_refresh=False, transient=True) as live:
            while True:

                key = readchar.readkey()
                if key == menu_controls["up"]:
                    self.update(-1)
                elif key == menu_controls["down"]:
                    self.update(1)
                elif key in menu_controls["select"]:
                    return self.index
                elif key == menu_controls["back"]:
                    return self.back()
                elif key == menu_controls["exit"]:
                    return sys.exit()

                live.update(self.render(), refresh=True)

    def empty(self) -> str:
        return "empty"

    def back(self) -> str:
        return "back"

def plot2d(values: torch.Tensor,
           cmap: str = "RdBu",
           title: str | None = None,
           x_title: str | None = None,
           y_title: str | None = None) -> None:

    print(values)
    fig, ax = plt.subplots()
    image = ax.imshow(values, cmap=cmap, aspect="auto")
    fig.colorbar(image, ax=ax)

    if title is not None:
        ax.set_title(title)
    if x_title is not None:
        ax.set_xlabel(x_title)
    if y_title is not None:
        ax.set_ylabel(y_title)

    plt.show()

def plot1d(values: torch.Tensor,
           cmap: str = "RdBu",
           title: str | None = None,
           x_title: str | None = None,
           y_title: str | None = None,
           vmin: float | None = None,
           vmax: float | None = None) -> None:

    line_color = plt.get_cmap(cmap)(0.85)

    fig, ax = plt.subplots()
    ax.plot(values, color=line_color, linewidth=2)

    if title is not None:
        ax.set_title(title)
    if x_title is not None:
        ax.set_xlabel(x_title)
    if y_title is not None:
        ax.set_ylabel(y_title)
    if vmin is not None:
        ax.set_ylim(bottom=vmin)
    if vmax is not None:
        ax.set_ylim(top=vmax)

    plt.show()

