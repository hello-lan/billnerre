
class PublisherORGExtractor:
    def __init__(self,
                 entity_key="text", 
                 label_key="label_name", 
                 label_value="发布者所属机构"):
        self.entity_key = entity_key
        self.label_key = label_key
        self.label_value = label_value

    def extract(self, text, labels):
        pub_org_labels = filter(lambda x: x[self.label_key]==self.label_value, labels)
        pub_orgs = list(map(lambda x:x[self.entity_key]), pub_org_labels)
        if len(pub_orgs) > 0:
            pub_org = max(pub_orgs, key=lambda x:len(x))   # 取字符长度最长的发布者所属机构
        else:
            pub_org = None
        return {self.label_value:pub_org}
    

class TradeDirExtractor:
    def __init__(self):
        pass
    
    def extract(self,text, labels):
        pass