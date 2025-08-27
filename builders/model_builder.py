from model.CMTNet import CMTNet

def build_model(model_name, num_classes):
    if model_name == 'CMTNet':
        return CMTNet(classes=num_classes)