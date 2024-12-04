from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from backend.commons.responses import ServiceResponse, ServiceResponseStatus
from backend.db.models.users import OAuthAccount, User  # type: ignore
from backend.logging import get_logger
from backend.services.commons.base import BaseService
from backend.services.google_oauth.service import get_user_info_from_google

logger = get_logger(__name__)


class GoogleAPIService(BaseService):
    def __init__(self, session: AsyncSession):
        self.session = session

    async def Read(self, user_email: str, user_id: UUID) -> ServiceResponse:
        try:
            # Create a SELECT statement to fetch the oauth_account by email.
            stmt = select(OAuthAccount).where(OAuthAccount.account_email == user_email)

            # Execute the statement and fetch one result.
            result = await self.session.execute(stmt)
            oauth_account = result.scalars().first()

            if oauth_account is None:
                return self.response(ServiceResponseStatus.NOTFOUND)
            user_info = await get_user_info_from_google(oauth_account.access_token)

            stmt_update = (
                update(User)
                .where(User.email == user_email)
                .values(userName=user_info["name"], userProfile=user_info["picture"])
            )
            await self.session.execute(stmt_update)
            await self.session.commit()
            user_info["userId"] = str(user_id)
            print(type(user_info))
            return self.response(ServiceResponseStatus.FETCHED, result=user_info)

        except SQLAlchemyError as e:
            logger.error(f"An error occurred: {e}")
            await self.session.rollback()
            return self.response(ServiceResponseStatus.ERROR)

    async def close(self) -> None:
        await self.session.close()
