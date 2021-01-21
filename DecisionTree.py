import pandas as pd
class DecisionTree:
    class Question:
        def __init__(self, column,value):
            self.column=column
            self.value=value
            
    class Node:
        def __init__(self,question,trueNode,falseNode,leafNode,prediction):
            self.question = question
            self.leafNode = leafNode
            self.trueNode = trueNode
            self.falseNode = falseNode
            self.prediction = prediction
        def print(self):
            print(self.question.column)

    def fit(self,x_train,y_train):
        data=x_train
        data["label"]=y_train
        gain, question=self.find_feature(data)
        leafNode=False
        predictions=None
        trueNode=None
        falseNode=None
        if gain==0:
            leafNode = True
            predictions = self.classCount(data)
        else:
            trueBranch,falseBranch=self.branchTree(question,data)
            trueNode = self.train(trueBranch)
            falseNode = self.train(falseBranch)
        self.rootNode=self.Node(question,trueNode,falseNode,leafNode,predictions)
        
    def train(self,data):
        gain, question=self.find_feature(data)
        leafNode=False
        predictions=None
        trueNode=None
        falseNode=None
        if gain==0:
            leafNode = True
            predictions = self.classCount(data)
        else:
            trueBranch,falseBranch=self.branchTree(question,data)
            trueNode = self.train(trueBranch)
            falseNode = self.train(falseBranch)
        return self.Node(question,trueNode,falseNode,leafNode,predictions)
    
    def classCount(self,data):
        p= data.groupby("label")["label"].count().to_dict()
        for key in p.keys():
            p[key]=(p[key]/len(data))
        return p
    
    def gini(self,data):
        counts = self.classCount(data)
        impurity = 1
        for lbl in counts:
            prob_of_lbl = counts[lbl] / float(len(data))
            impurity -= prob_of_lbl**2
        return impurity
    
    def info_gain(self,left, right, current_uncertainty):
        p = float(len(left)) / (len(left) + len(right))
        return current_uncertainty - p * self.gini(left) - (1 - p) * self.gini(right)
    
    def find_feature(self,data):
        gain = 0
        question = None
        current_uncertainty = self.gini(data)
        for col in data.drop("label",axis=1):
            values=data[col].unique()
            for val in values:
                q = self.Question(col,val)
                trueBranch,falseBranch=self.branchTree(q,data)
                if len(trueBranch)==0 or len(falseBranch)==0:
                    continue
                g = self.info_gain(trueBranch, falseBranch, current_uncertainty)
                if g >= gain:
                    gain, question = g, q
        return gain,question
    
    def branchTree(self,question,data):
        trueBranch = data[data[question.column]==question.value]
        falseBranch = data[data[question.column]!=question.value]
        return trueBranch,falseBranch
    
    def print(self):
        print("{}->{}".format(self.rootNode.question.column,self.rootNode.question.value))
        self.print_tree(self.rootNode.trueNode,2,"TrueNode: ")
        self.print_tree(self.rootNode.falseNode,2,"FalseNode: ")
    def print_tree(self,node,space,name):
        print("      "*space+name,end="")
        if node.leafNode:
            print(node.prediction)
            return
        print("{}->{}".format(node.question.column,self.rootNode.question.value))
        self.print_tree(node.trueNode,space+1,"TrueNode: ")
        self.print_tree(node.falseNode,space+1,"FalseNode: ")
        
    def predict(self,data,probability=False):
        if isinstance(data,pd.Series):
            data=data.to_frame().T
        result=[]
        for row in data.iterrows():
            row=row[1]
            node=self.rootNode
            while not node.leafNode:
                if row[node.question.column]==node.question.value:
                    node=node.trueNode
                else:
                    node=node.falseNode
            if probability:
                result.append(node.prediction)
            else:
                result.append(max(node.prediction, key=node.prediction.get))
        return result