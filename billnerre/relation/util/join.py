
def join(entities, prefix='',sufix=''):
    """将多个具有相同标签的实体拼接得到一个正则表达式
    """
    entities = sorted(entities,key=lambda x:len(x), reverse=True)
    _entities = [prefix + ent + sufix for ent in entities]
    express = "|".join(_entities)
    return express