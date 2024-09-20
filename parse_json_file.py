import json

def main(file_name):
    with open(file_name, 'r') as f:
        data = json.load(f)



if __name__ == '__main__':
    main()
