from models.BaseModel import BaseModel

model = BaseModel(num_classes=2, freeze_bert=False)

print(model)
