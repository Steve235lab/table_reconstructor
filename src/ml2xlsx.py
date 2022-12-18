import pandas as pd


class ML2Xlsx:
    def __init__(self, ml_file_path: str):
        # 读取标记语言文件
        self.ml_file_path = ml_file_path
        # TableBank中用来表示表格的标记语言标签集合
        self.mark_set = {'<tabular>', '</tabular>', '<thead>', '</thead>', '<tbody>', '</tbody>',
                         '<tr>', '</tr>', '<tdy>', '</tdy>', '<tdn>', '</tdn>'}
        self.output_path = '../output/ML2Xlsx/'

    def ml2xlsx(self):
        ml_file = open(self.ml_file_path, 'r', encoding='utf-8')
        table_cnt = 0
        while True:
            try:
                ml_line = ml_file.readline()
            except:
                break
            if ml_line:
                table_df = pd.DataFrame()
                new_line = pd.DataFrame()
                new_cell = pd.DataFrame()
                for mark in ml_line.split(' '):
                    # print(mark)
                    # if mark not in self.mark_set:
                    #     print("Undefined Mark: " + mark)
                    #     break
                    if mark == '</tabular>':
                        break
                    elif mark == '<tr>':
                        new_line = pd.DataFrame()
                    elif mark == '</tr>':
                        table_df = pd.concat([table_df, new_line], ignore_index=True, axis=0)
                    elif mark == '<tdy>':
                        new_cell = pd.DataFrame(['Non-empty Cell'])
                        new_line = pd.concat([new_line, new_cell], ignore_index=True, axis=1)
                    elif mark == '<tdn>':
                        new_cell = pd.DataFrame(['Empty Cell'])
                        new_line = pd.concat([new_line, new_cell], ignore_index=True, axis=1)
                print(table_df)
                table_df.to_excel(self.output_path + str(table_cnt) + '.xlsx',
                                  sheet_name='Sheet1', index=False, header=False)
            else:
                break
            table_cnt += 1
        ml_file.close()


if __name__ == "__main__":
    m2x = ML2Xlsx('../output/pred.txt')
    m2x.ml2xlsx()
