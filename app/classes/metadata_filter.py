from typing import Literal, Self, Annotated, Any

from pydantic import BaseModel, Tag, Discriminator

from app.types import Metadata, ExistingMetadata, MetadataValue


class NestedConditionBlock(BaseModel):
    """This object is responsible for the proper evaluation of conditions and can contain itself."""

    condition_block_type: Literal["nested"]
    block_operator: Literal["and", "or"]
    blocks: list["ConditionBlockItem"]

    def calculate_condition_block(
        self: Self,
        metadata: Metadata,
    ) -> bool:
        """This function calculates the nested condition block."""
        # We return True if the list of the blocks are empty.
        if not self.blocks:
            return True

        # We return False if blocks exists and the Metadata is not existing.
        if self.blocks and not metadata:
            return False

        # If the block operator is 'and', we return False if one of the blocks is False or None.
        #   We return True if all the blocks are True.
        if self.block_operator == "and" and self.blocks:
            for and_block in self.blocks:
                if not and_block.calculate_condition_block(metadata=metadata):
                    return False
            return True
        # If the block operator is 'or', we return True if one of the blocks is True.
        #   We return False if all the blocks are False or None.
        if self.block_operator == "or":
            for or_block in self.blocks:
                if or_block.calculate_condition_block(metadata=metadata):
                    return True
        return False


class EqualConditionBlock(BaseModel):
    """This object describes the structure of an equal comparison."""

    condition_block_type: Literal["single"]
    comparator_name: Literal["equal"]
    meta_field_name: str
    expected_value: Any

    def calculate_condition_block(self: Self, metadata: ExistingMetadata) -> bool:
        """This function calculates the condition block."""
        return metadata.get(self.meta_field_name) == self.expected_value


class NotEqualConditionBlock(BaseModel):
    """This object describes the structure of a not equal comparison."""

    condition_block_type: Literal["single"]
    comparator_name: Literal["not_equal"]
    meta_field_name: str
    expected_value: Any

    def calculate_condition_block(self: Self, metadata: ExistingMetadata) -> bool:
        """This function calculates the condition block."""
        return metadata.get(self.meta_field_name) != self.expected_value


class GreaterThanConditionBlock(BaseModel):
    """This object describes the structure of a greater than comparison."""

    condition_block_type: Literal["single"]
    comparator_name: Literal["greater_than"]
    meta_field_name: str
    expected_value: Any

    def calculate_condition_block(self: Self, metadata: ExistingMetadata) -> bool:
        """This function calculates the condition block."""
        meta_value = metadata.get(self.meta_field_name)
        if meta_value is not None:
            return self.expected_value < meta_value
        return False


class GreaterThanEqualConditionBlock(BaseModel):
    """This object describes the structure of a greater than or equal comparison."""

    condition_block_type: Literal["single"]
    comparator_name: Literal["greater_than_equal"]
    meta_field_name: str
    expected_value: Any

    def calculate_condition_block(self: Self, metadata: ExistingMetadata) -> bool:
        """This function calculates the condition block."""
        meta_value = metadata.get(self.meta_field_name)
        if meta_value is not None:
            return self.expected_value <= meta_value
        return False


class LowerThanConditionBlock(BaseModel):
    """This object describes the structure of a lower than comparison."""

    condition_block_type: Literal["single"]
    comparator_name: Literal["lower_than"]
    meta_field_name: str
    expected_value: Any

    def calculate_condition_block(self: Self, metadata: ExistingMetadata) -> bool:
        """This function calculates the condition block."""
        meta_value = metadata.get(self.meta_field_name)
        if meta_value is not None:
            return self.expected_value > meta_value
        return False


class LowerThanEqualConditionBlock(BaseModel):
    """This object describes the structure of a lower than or equal comparison."""

    condition_block_type: Literal["single"]
    comparator_name: Literal["lower_than_equal"]
    meta_field_name: str
    expected_value: Any

    def calculate_condition_block(self: Self, metadata: ExistingMetadata) -> bool:
        """This function calculates the condition block."""
        meta_value = metadata.get(self.meta_field_name)
        if meta_value is not None:
            return self.expected_value >= meta_value
        return False


class ContainmentConditionBlock(BaseModel):
    """Shared base of the comparators that apply `in` to a metadata value."""

    condition_block_type: Literal["single"]
    # Narrowed to its own Literal by each subclass; declared here so the shared guard can name it.
    comparator_name: str
    meta_field_name: str
    expected_value: Any

    def ensure_container(self: Self, meta_value: MetadataValue) -> None:
        """Raise on a scalar field. A silent False is indistinguishable from an honest no-match."""
        # bool is excluded despite subclassing int, because `x in True` raises rather than False.
        if not isinstance(meta_value, (str, list)):
            exception_message = (
                f"'{self.comparator_name}' needs a list or string metadata field, but "
                f"'{self.meta_field_name}' holds {type(meta_value).__name__} ({meta_value!r}). "
                f"Use 'equal'/'not_equal' for a scalar, or a range comparator for a number."
            )
            raise TypeError(exception_message)


class ContainsConditionBlock(ContainmentConditionBlock):
    """This object describes the structure of a contains comparison."""

    comparator_name: Literal["contains"]

    def calculate_condition_block(self: Self, metadata: ExistingMetadata) -> bool:
        """This function calculates the condition block."""
        meta_value = metadata.get(self.meta_field_name)
        if meta_value is not None:
            self.ensure_container(meta_value)
            return self.expected_value in meta_value
        return False


class NotContainsConditionBlock(ContainmentConditionBlock):
    """This object describes the structure of a not contains comparison."""

    comparator_name: Literal["not_contains"]

    def calculate_condition_block(self: Self, metadata: ExistingMetadata) -> bool:
        """This function calculates the condition block."""
        meta_value = metadata.get(self.meta_field_name)
        if meta_value is not None:
            self.ensure_container(meta_value)
            return self.expected_value not in meta_value
        return False


def get_condition_block_discriminator_value(checked_value: Any) -> str:
    """This function returns the discriminator value of the condition block."""
    if isinstance(checked_value, dict):
        condition_block_type = checked_value.get("condition_block_type", None)
        comparator_name = checked_value.get("comparator_name", "")
    else:
        condition_block_type = getattr(checked_value, "condition_block_type", None)
        comparator_name = getattr(checked_value, "comparator_name", "")
    # If we have a comparator_name, we return the discriminator value with it.
    return f"{condition_block_type}_{comparator_name}" if comparator_name else str(condition_block_type)


ConditionBlockItem = Annotated[
    Annotated[NestedConditionBlock, Tag("nested")]
    | Annotated[EqualConditionBlock, Tag("single_equal")]
    | Annotated[GreaterThanConditionBlock, Tag("single_greater_than")]
    | Annotated[GreaterThanEqualConditionBlock, Tag("single_greater_than_equal")]
    | Annotated[LowerThanConditionBlock, Tag("single_lower_than")]
    | Annotated[LowerThanEqualConditionBlock, Tag("single_lower_than_equal")]
    | Annotated[ContainsConditionBlock, Tag("single_contains")]
    | Annotated[NotContainsConditionBlock, Tag("single_not_contains")]
    | Annotated[NotEqualConditionBlock, Tag("single_not_equal")],
    Discriminator(get_condition_block_discriminator_value),
]

# We need to rebuild the model, to work with the latter defined ConditionBlockItem.
NestedConditionBlock.model_rebuild()

MetaDataFilter = ConditionBlockItem

EMPTY_METADATA_FILTER = NestedConditionBlock(condition_block_type="nested", block_operator="and", blocks=[])
