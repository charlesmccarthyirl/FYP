from BreadAndButter import create_named_data_set_generators

named_data_sets = create_named_data_set_generators((
    ('iris.pcdb.gz', {'sc': 20}),
    ('wine.pcdb.gz', {'sc': 25}),
    ('zoo.pcdb.gz', {'sc': 25}),
    ('dermatology.pcdb.gz', {'sc': 40}),
    ('hepatitis.pcdb.gz', {'sc': 10}),
    ('breathalyzer.pcdb.gz', {'sc': 50}),
    ('haberman.pcdb.gz', {'sc': 40}),
    ('glass.pcdb.gz', {'sc': 40}),
    ('tae.pcdb.gz', {'sc': 30}),
    ('heartdisease.pcdb.gz', {'sc': 30}),
    ('lungcancer.pcdb.gz', {'sc': 10}),
    ('WinXwin.pcdb.gz', {'sc': 150}),
    ('Comp.pcdb.gz', {'sc': 150}),
    ('Talk.pcdb.gz', {'sc': 150}),
    ('Vehicle.pcdb.gz', {'sc': 150}),
))