from pydantic import BaseModel, computed_field


class CategoryData(BaseModel):
    categories: list[str] = []
    category_lines: dict[str, list[str]] = {}

    @computed_field  # type: ignore[misc]
    @property
    def n_categories(self) -> int:
        return len(self.categories)
