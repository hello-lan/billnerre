#coding:utf8
import torch as t
import time
import inspect


class BasicModule(t.nn.Module):
    """
    封装了nn.Module,主要是提供了save和load两个方法
    """

    def __init__(self):
        super(BasicModule,self).__init__()
        self.model_name = self.__class__.__name__

    def load(self, path):
        """
        可加载指定路径的模型
        """
        self.load_state_dict(t.load(path))

    def save(self, name=None):
        """
        保存模型，默认使用“模型名字+时间”作为文件名
        """
        if name is None:
            prefix = 'checkpoints/' + self.model_name + '_best_'
            name = time.strftime(prefix + '%Y%m%d.pth')
        t.save(self.state_dict(), name)
        return name

    def get_optimizer(self, lr, weight_decay):
        # return t.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        params = [p for p in self.parameters() if p.requires_grad]
        return t.optim.Adam(params, lr=lr, weight_decay=weight_decay)



class BasicModuleV2(t.nn.Module):
    """
    封装了nn.Module,主要是提供了save和load_model两个方法
    """
    def __new__(cls, *args, **kwargs):
        obj = super(t.nn.Module, cls).__new__(cls)
        obj._init_args = args
        obj._init_kwargs = kwargs
        return obj

    def __init__(self):
        super(BasicModuleV2,self).__init__()
        self.model_name = self.__class__.__name__

    def save(self, path=None,word2id=None):
        """
        保存模型
        """
        if path is None:
            dt = time.strftime('%Y%m%d')
            path = 'checkpoints/%s_best_%s.pth'%(self.model_name, dt)
        init_kwargs = inspect.getcallargs(self.__init__, *self._init_args, **self._init_kwargs)
        
        meta = dict(init_kwargs=init_kwargs, model_state=self.state_dict(),word2id=word2id)
        t.save(meta, path)
        return path

    def get_optimizer(self, lr, weight_decay=0):
        # return t.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        params = [p for p in self.parameters() if p.requires_grad]
        return t.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    
    @classmethod
    def load_model(cls,path):
        meta = t.load(path)
        init_kwargs = meta["init_kwargs"]
        # print(init_kwargs)
        if "self" in init_kwargs:
            init_kwargs.pop("self")
        model = cls(**init_kwargs)
        model.load_state_dict(meta['model_state'])
        word2id = meta.get('word2id')
        return model, word2id