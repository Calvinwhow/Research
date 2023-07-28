from file_writer import FileWriter

def main():
    filename = 'hello.txt'
    text = 'Hello, world!'
    append = True

    FileWriter.write_to_file(filename, text, append)

if __name__ == '__main__':
    main()
