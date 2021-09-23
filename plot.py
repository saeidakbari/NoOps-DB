import argparse

import matplotlib.pyplot as plt
import pandas as pd

from modeling.type import Target


class Plotter:
    """
    Plotter that plots OUModel training results based on CSV output file
    """

    RESULT_PATH = ""

    def __init__(self, res_path, split, type) -> None:
        self.RESULT_PATH = res_path
        self.split = split
        self.type = type
        pass

    def __lables(self) -> list:
        return [f.name for f in Target if f > 1]

    def __loadResultFile(self) -> pd.DataFrame:
        self.DATA = pd.read_csv(r"{}/ou_runner.csv".format(self.RESULT_PATH))

    def splitBare(self):
        self.DATA = self.DATA[self.DATA['Method'].str.contains(
            "transform") == True]

    def splitTransformed(self):
        self.DATA = self.DATA[self.DATA['Method'].str.contains(
            "transform") == False]

    def calcOU(self) -> pd.DataFrame:
        self.__loadResultFile()
        if self.split == "bare":
            self.splitTransformed()
        elif self.split == "transform":
            self.splitBare()

        data = self.DATA
        data["AverageOU"] = pd.DataFrame(
            data, columns=self.__lables()).mean(axis=1)
        return data

    def plotErrPerMethod(self):
        ous = self.calcOU()
        df = ous.pivot(index="OpUnit", columns="Method", values='AverageOU')
        df.plot(kind='bar', figsize=(15, 5), color=[
                'red', 'blue', '#e37827', '#275444'])

        plt.xlabel('Op Unit')
        plt.ylabel('Error')
        plt.title('Average error per method')

    def do(self):
        if self.type == "ou":
            self.plotErrPerMethod()
        elif self.type == "":
            pass
        else:
            self.plotErrPerMethod()
        plt.show()


# ==============================================
# main
# ==============================================
if __name__ == '__main__':
    aparser = argparse.ArgumentParser(description="Model Trainer")
    aparser.add_argument('--input_path', default='data/trained_model_metric_results',
                         help='Input file for plotting the result')
    aparser.add_argument('--split', default='bare',
                         help='split the transformed result')
    aparser.add_argument('--type', default='all',
                         help='show result based on')
    args = aparser.parse_args()

    plotter = Plotter(args.input_path, args.split, args.type)
    plotter.do()
