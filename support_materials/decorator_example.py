from abc import ABC, abstractmethod
from enum import Enum

import pandas as pd

APPLIED_CHECKS: list["CheckInterface"] = []


def register_check(_class):
    """
    The idea is to be able to clearly see that check is used/not used.
    And to be able to easily switch on/off.
    Checks were migrated from legacy system and it wasn't clear which checks were actually needed.
    """
    APPLIED_CHECKS.append(_class)
    return _class


class CheckScope(Enum):
    """Defines the scope of information required for the check to be run"""
    WELL = 'WELL'
    PLATE = 'PLATE'


class CheckInterface(ABC):
    message: str
    scope: CheckScope
    id_columns: list[str]

    @classmethod
    @abstractmethod
    def _check(cls, props_df: pd.DataFrame) -> pd.Series:
        """ Return pd.Series with True values, if check is failed, False if passed"""
        pass

    @classmethod
    def check(cls, props_df: pd.DataFrame) -> pd.DataFrame:
        """ Return pd.DataFrame with warning scope, message, and id_columns for a single check """
        props_df = props_df.copy()
        props_df = props_df.loc[cls._check, cls.id_columns]
        props_df['scope'] = cls.scope.value
        props_df['message'] = cls.message
        props_df.drop_duplicates(inplace=True)
        return props_df


class WellCheck(CheckInterface):
    scope = CheckScope.WELL
    id_columns = ('plate_id', 'well_row', 'well_column')


class PlateCheck(CheckInterface):
    scope = CheckScope.PLATE
    id_columns = ('plate_id',)


@register_check
class CheckIsMissingCompoundConc(WellCheck):
    """Check `compound_id` is set where `concentration` is not set"""
    message = "`compound_id` is set without `concentration`"

    @classmethod
    def _check(cls, props_df: pd.DataFrame) -> pd.Series:
        return (
            props_df["compound_id"].notnull()
            & props_df["concentration"].isnull()
        )


@register_check
class CheckEmptyPlate(PlateCheck):
    message = "All wells in plate have only 'N' values for 'cells_plated' column"

    @classmethod
    def _check(cls, props_df: pd.DataFrame) -> pd.Series:
        groups = []
        for _plate_id, group in props_df.groupby(["plate_id"]):
            group["check_column"] = all(group["cells_plated"] == "N")
            groups.append(group)

        return pd.concat(groups)["check_column"]


class PlateValidationTask:
    def __init__(self, plate_df):
        self.plate_df = plate_df

    def validate(self) -> pd.DataFrame:
        """ Perform checks and return DataFrame with warnings """
        check_warnings = [check.check(self.plate_df) for check in APPLIED_CHECKS]
        warnings_df = pd.concat(check_warnings, ignore_index=True)
        # Reorder columns so that order is consistent
        # (it will insert column with NANs if it wasn't present)
        warnings_df = warnings_df.reindex(
            [
                'scope',
                'message',
                'well_row',
                'well_column',
                'plate_id',
            ],
            axis=1
        )
        warnings_df.sort_values(
            by='scope',
            key=lambda col: col.map({
                CheckScope.PLATE.value: 1,
                CheckScope.WELL.value: 2,
            }),
            inplace=True
        )
        return warnings_df


plate_df = pd.DataFrame([
    {
        'cells_plated': 'Y',
        'compound_id': 1,
        'concentration': 1,
        'plate_id': 1,
        'well_row': 1,
        'well_column': 1,
    },
    {
        'cells_plated': 'Y',
        'compound_id': 1,
        'concentration': None,
        'plate_id': 1,
        'well_row': 1,
        'well_column': 1,
    }
])
task = PlateValidationTask(plate_df)
warnings_df = task.validate()
print(warnings_df)


plate_df = pd.DataFrame([
    {
        'cells_plated': 'N',
        'compound_id': None,
        'concentration': None,
        'plate_id': 1,
        'well_row': 1,
        'well_column': 1,
    },
    {
        'cells_plated': 'N',
        'compound_id': None,
        'concentration': None,
        'plate_id': 1,
        'well_row': 1,
        'well_column': 1,
    }
])
task = PlateValidationTask(plate_df)
warnings_df = task.validate()
print(warnings_df)


plate_df = pd.DataFrame([
    {
        'cells_plated': 'Y',
        'compound_id': 1,
        'concentration': 1,
        'plate_id': 1,
        'well_row': 1,
        'well_column': 1,
    }
])
task = PlateValidationTask(plate_df)
warnings_df = task.validate()
print(warnings_df)