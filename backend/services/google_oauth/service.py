from typing import Any, Dict

import httpx
from fastapi import HTTPException


async def get_user_info_from_google(access_token: str) -> Dict[str, Any]:
    userinfo_url = "https://openidconnect.googleapis.com/v1/userinfo"
    headers = {"Authorization": f"Bearer {access_token}"}

    async with httpx.AsyncClient() as client:
        userinfo_response = await client.get(userinfo_url, headers=headers)
        if userinfo_response.status_code != 200:
            raise HTTPException(
                status_code=userinfo_response.status_code,
                detail="Error fetching user info",
            )

        return userinfo_response.json()
