from fastapi import APIRouter

from yoga_app.api import content

router = APIRouter()

router.include_router(content.router)
