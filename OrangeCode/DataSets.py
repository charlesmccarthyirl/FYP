from BreadAndButter import create_named_data_set_generators

named_data_sets = create_named_data_set_generators((
    ('iris.pcdb.gz', {'sc': 20}),
    ('dermatology.pcdb.gz', {'sc': 40}),
    ('wine.pcdb.gz', {'sc': 25}),
    ('zoo.pcdb.gz', {'sc': 25}),
    ('hepatitis.pcdb.gz', {'sc': 10}),
    ('breathalyzer.pcdb.gz', {'sc': 50}),
    ('haberman.pcdb.gz', {'sc': 40}),
    ('glass.pcdb.gz', {'sc': 40}),
    ('tae.pcdb.gz', {'sc': 30}),
    ('heart_disease.pcdb.gz', {'sc': 30}),
    ('lung-cancer.pcdb.gz', {'sc': 10}),
    ('pima-indians-diabetes.pcdb.gz', {'sc': 60}),
))