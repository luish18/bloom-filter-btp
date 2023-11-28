from bloom_filter_btp.img_processing import capture_image, extract_face


def register_user() -> None:
    # get user name
    username = input("Enter your name: ")

    print("Please, look at the camera...")
    img = capture_image()

    face = extract_face(img)

    print("Extracting face features...")

    print("Generating user BF...")

    print("Saving user BF...")

    print("User registered successfully!")


def check_user_in_db(threshold: float) -> bool:
    print("Please, look at the camera...")
    img = capture_image()

    face = extract_face(img)

    print("Extracting face features...")

    print("Generating user BF...")

    print("Loading all BF templates...")
    bf_templates = []

    print("Checking if user is in the database...")

    print("User not found!")
    return False


def main() -> None:
    while True:
        print("1. Register user")
        print("2. Login user")
        print("3. Exit")
        option = input("Select an option: ")

        if option == "1":
            register_user()
        elif option == "2":
            check_user_in_db(0.5)
        elif option == "3":
            print("Bye!")
            break
        else:
            print("Invalid option")
            continue


if __name__ == "__main__":
    main()
