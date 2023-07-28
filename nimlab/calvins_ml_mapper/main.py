# main.py

from utils.cli_resource_printer import print_text_file
import json

def menu_iterator(menu_json):
    user_data = {}
    current_step = 1
    max_step = len(menu_json)

    while current_step <= max_step:
        print(menu_json[str(current_step)]["label"])
        user_input = input()

        if user_input.lower() == "back":
            current_step = max(1, current_step - 1)
        else:
            function_name = menu_json[str(current_step)]["functions"][0]
            user_data[function_name] = user_input
            current_step += 1

    return user_data

def main():
    with open("menus/voxelwise_ml_mapper/menu.json", "r") as f:
        menu_structure = json.load(f)

    user_data = menu_iterator(menu_structure)
    print("Collected user data:", user_data)

if __name__ == "__main__":
    print_text_file('resources/home_screen.txt')
    main()