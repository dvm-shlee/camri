import re

def strip_columnname(column_name):
    'this function strip levels from the column name of category factors'
    ptrn = r'(C\([^\)]+\))(\[[^\]]+])'
    if ':' in column_name:
        columns = []
        for c in column_name .split(':'):
            matched = re.match(ptrn, c)
            if matched != None:
                term, cond = matched.groups()
                columns.append(strip_termname(term) + cond)
            else:
                columns.append(c)
        return ':'.join(columns)
    else:
        matched = re.match(ptrn, column_name)
        if matched != None:
            term, cond = matched.groups()
            return strip_termname(term) + cond
        else:
            return column_name

            
def strip_termname(term_name):
    'this function strip levels from the term name of category factors'
    ptrn = r'C\(([^,]+), .*\)'
    if ':' in term_name:
        terms = []
        for t in term_name.split(':'):
            matched = re.match(ptrn, t)
            if matched != None:
                terms.append(f'{matched.groups()[0]}')
            else:
                terms.append(t)
        return ':'.join(terms)
    else:
        matched = re.match(ptrn, term_name)
        if matched != None:
            return f'{matched.groups()[0]}'
        else:
            return term_name
        
        
def lrange(*args, **kwargs):
    return list(range(*args, **kwargs))