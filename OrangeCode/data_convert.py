from pcdcd_converter import main
import os

data_files = [
              'iris.tab',
              'dermatology.csv',
              'wine.tab',
              'zoo.tab',
              'hepatitis.csv',
              'breathalyzer.csv',
              'haberman.csv',
              'glass.tab',
              'tae.csv',
              'heartdisease.tab',
              'lungcancer.tab'
              ]
data_dir = '../Datasets'

for df in data_files:
    base_name = os.path.splitext(os.path.basename(df))[0]
    
    input_file = os.path.join(data_dir, df)
    output_file = os.path.join(data_dir, base_name + ".pcdb")
    
    main(None, input_file, output_file, 'Euclidean', 'proto')
