import os, re, pickle, sys, argparse

def get_args_review():
    parser = argparse.ArgumentParser()
    parser.add_argument('-REV', default="TODO", type=str, help='REV or TODO')
    args = parser.parse_args()
    return args


def fetch_line(dir, filename, REV):
    file = open(os.path.join(dir,filename))
    if REV =="REV":
        check_str = "#REV"
    elif REV =="TODO":
        check_str = "#TODO"

    for line_no, line in enumerate(file.readlines()):
        check_line = re.sub(r'\s+', '', line)


        if check_line.startswith(check_str):
            print(" file name is: {}".format(filename))
            print("line number: {}".format(line_no))
            print("code line:")
            print(line)

if __name__ == '__main__':
    args = get_args_review()
    directory = os.getcwd()
    for subdirs, dirs, files in os.walk(directory):

        for file in files:
            filename = os.fsdecode(file)
            if filename.endswith(".py"):
                fetch_line(directory, filename, args.REV)
        for dir in dirs:
            new_dir = os.path.join(directory, dir)
            for file in os.listdir(new_dir):
                filename = os.fsdecode(file)
                if filename.endswith(".py"):
                    fetch_line(new_dir, filename, args.REV)
        

    


