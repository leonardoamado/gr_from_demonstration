from recognizer import Recognizer

#Define datasets here
#Each element of this list is a goal recognition problem with 5 different observabilities
BLOCKS = ['output/blocks_gr/']
def run_experiments():
    recog = Recognizer()
    for folder in BLOCKS:
        recog.complete_recognition_folder(folder)

if __name__ == "__main__":
    run_experiments()