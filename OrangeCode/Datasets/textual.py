from ..BreadAndButter import create_named_data_set_generators

named_data_sets = create_named_data_set_generators((
    ('WinXwin.pcdb.gz', {'sc': 150}),
    ('Comp.pcdb.gz', {'sc': 150}),
    ('Talk.pcdb.gz', {'sc': 150}),
    ('Vehicle.pcdb.gz', {'sc': 150}),
))