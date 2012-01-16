from BreadAndButter import create_named_data_set_generators

named_data_sets = create_named_data_set_generators((
    ('iris.pcdb', {'sc': 20}),
    ('dermatology.pcdb', {'sc': 40}),
    ('wine.pcdb', {'sc': 25}),
    ('zoo.pcdb', {'sc': 25}),
    ('hepatitis.pcdb', {'sc': 10}),
    ('breathalyzer.pcdb', {'sc': 50}),
    ('haberman.pcdb', {'sc': 40}),
    ('glass.pcdb', {'sc': 40}),
    ('tae.pcdb', {'sc': 30}),
    ('heart_disease.pcdb', {'sc': 30}),
    ('lung-cancer.pcdb', {'sc': 10}),
    ('pima-indians-diabetes.pcdb', {'sc': 60}),
))