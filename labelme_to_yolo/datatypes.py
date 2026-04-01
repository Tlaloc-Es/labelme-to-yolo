from typing import Annotated

from pydantic import BaseModel, Field

Point = Annotated[list[float], Field(min_length=2, max_length=2)]


class LabelMeShape(BaseModel):
    label: str
    points: list[Point]
    shape_type: str


class LabelMe(BaseModel):
    shapes: list[LabelMeShape]
    image_height: int
    image_width: int


class OutputPaths(BaseModel):
    dataset_path: str
    images_path: str
    images_train_path: str
    images_val_path: str
    images_test_path: str
    labels_path: str
    labels_train_path: str
    labels_val_path: str
    labels_test_path: str


class FileNameAndExtension(BaseModel):
    file_name: str
    extension: str


class Polygon(BaseModel):
    points: list[float]
    label_index: int
    label_name: str

    def get_representation(self):
        return f"{self.label_index} {' '.join(map(str, self.points))}"


class ShapeProcessed(BaseModel):
    path: str
    file_name: str
    polygons: list[Polygon]


class ShapesProcessed(BaseModel):
    shapes: list[ShapeProcessed] = []


class YoloYML(BaseModel):
    train: str
    val: str
    test: str
    nc: int
    names: list[str]
