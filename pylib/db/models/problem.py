from typing import TypedDict, Optional
from datetime import datetime
from bson import ObjectId


class SavedProblem(TypedDict):
    name: str                            # user-defined label for this saved problem
    created_time: datetime
    updated_time: datetime
    draft: dict                          # full ProblemDraft JSON
    background_image_id: Optional[ObjectId]  # GridFS reference, None if no image
    image_world_bounds: Optional[list]   # [xmin, ymin, xmax, ymax] or None
