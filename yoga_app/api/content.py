
import shutil
from typing import List

from fastapi import APIRouter, Depends, File, UploadFile
from yoga_app.ml.core import refactor_this_later
from uuid import uuid4
import os

router = APIRouter(prefix="/content", tags=["classify Pose"])


@router.post("/classify_pose")
async def _classify_pose(
    image: UploadFile = File(...),
):
    """
    Classify a pose from an image.
    save the image to a file named image.jpg and then call the classify_pose function

    """
    path_name = f"{str(uuid4())}.jpg"
    try:
       
        # save the image to a file named image.jpg
        with open(path_name, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        # call the classify_pose function
        suggestions = refactor_this_later("image.jpg")
        return {"suggestions": suggestions}
    finally:
        os.remove(path_name)

