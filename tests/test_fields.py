import pytest

from mizan.fields import NoSuchField, check_expected, differences, lookup, split_path

MONEY = {"amount": "decimal", "currency": "exact"}


@pytest.mark.parametrize(
    ("kind", "expected", "read"),
    [
        ("decimal", "12.50", "12.5"),
        ("decimal", "12.50", 12.5),
        ("decimal", "1250.00", 1250),
        ("date", "2026-10-03", "03 Oct 2026"),
        ("date", "2026-10-03", "3 October 2026"),
        ("date", "2026-10-03", "Oct 3, 2026"),
        ("date", "2026-10-03", "October 3, 2026"),
        ("text", "Corner  Market", " corner market\n"),
        ("exact", True, True),
        ("exact", "CAD", "CAD"),
        ("exact", {"a": 1, "b": [True]}, {"b": [True], "a": 1}),
        (MONEY, {"amount": "11.79", "currency": "CAD"}, {"amount": 11.790, "currency": "CAD"}),
        (MONEY, {"amount": "11.79"}, {"amount": "11.79", "currency": "USD"}),
    ],
)
def test_values_that_mean_the_same_thing_are_equal(kind, expected, read):
    assert differences(kind, expected, read) is None


@pytest.mark.parametrize(
    ("kind", "expected", "read", "reason"),
    [
        ("decimal", "12.5", "12.51", 'expected "12.5", read "12.51"'),
        ("date", "03 Oct 2026", "2026-10-04", 'expected "03 Oct 2026", read "2026-10-04"'),
        ("text", "Corner Market", "Corner Mart", 'expected "Corner Market", read "Corner Mart"'),
        ("exact", True, 1, "expected true, read 1"),
        ("exact", 1, True, "expected 1, read true"),
        ("exact", {"flag": True}, {"flag": 1}, 'expected {"flag": true}, read {"flag": 1}'),
        ("exact", [True], [1], "expected [true], read [1]"),
        ("exact", "CAD", "cad", 'expected "CAD", read "cad"'),
        ("decimal", "12.50", "12.5x", 'expected "12.50", read "12.5x" (not a decimal)'),
        ("decimal", "1", True, 'expected "1", read true (not a decimal)'),
        (
            "date",
            "2026-10-03",
            "03/10/2026",
            'expected "2026-10-03", read "03/10/2026" (not a date)',
        ),
    ],
)
def test_values_that_differ_give_the_reason(kind, expected, read, reason):
    assert differences(kind, expected, read) == reason


def test_money_differs_by_amount_and_by_the_exact_currency():
    expected = {"amount": "1250.00", "currency": "CAD"}

    assert differences(MONEY, expected, {"amount": "1", "currency": "cad"}) == (
        'amount: expected "1250.00", read "1"; currency: expected "CAD", read "cad"'
    )
    assert differences(MONEY, expected, {"amount": "1250"}) == "currency: missing"
    assert differences(MONEY, expected, "1250.00 CAD") == 'expected an object, read "1250.00 CAD"'


def test_an_expected_null_matches_only_a_null():
    assert differences("decimal", None, None) is None
    assert differences("decimal", None, "0.00") == 'expected null, read "0.00"'
    assert differences("decimal", "0.00", None) == 'expected "0.00", read null'


def test_a_null_inside_money_follows_the_same_rule_per_key():
    expected = {"amount": "11.79", "currency": None}

    assert differences(MONEY, expected, {"amount": "11.79", "currency": None}) is None
    assert differences(MONEY, expected, {"amount": "11.79", "currency": "CAD"}) == (
        'currency: expected null, read "CAD"'
    )


@pytest.mark.parametrize(
    ("kind", "expected", "message"),
    [
        ("date", "03/10/2026", 'expected "03/10/2026" is not a date'),
        ("decimal", "NaN", 'expected "NaN" is not a decimal'),
        ("decimal", "Infinity", 'expected "Infinity" is not a decimal'),
        ("decimal", True, "expected true is not a decimal"),
        ("text", 77955, "expected 77955 is not text"),
        (MONEY, "11.79", 'expected "11.79" is not an object'),
        (MONEY, {"amount": "eleven"}, 'expected "eleven" is not a decimal'),
        (MONEY, {"amount": "1", "tax": "0"}, 'expected has the key "tax"'),
    ],
)
def test_an_expected_value_that_cannot_be_compared_is_refused(kind, expected, message):
    with pytest.raises(ValueError, match=message):
        check_expected(kind, expected)


def test_an_expected_null_fits_every_kind():
    for kind in ("text", "decimal", "date", "exact", MONEY):
        check_expected(kind, None)


def test_a_path_descends_through_objects_and_finds_a_present_null():
    output = {"total": {"amount": "12.5", "currency": None}}

    assert lookup(output, split_path("total.amount")) == "12.5"
    assert lookup(output, split_path("total.currency")) is None


@pytest.mark.parametrize("path", ["tip", "total.tax", "total.amount.value"])
def test_a_path_that_leads_nowhere_is_no_field_and_not_a_null(path):
    with pytest.raises(NoSuchField):
        lookup({"total": {"amount": "12.5"}}, split_path(path))


@pytest.mark.parametrize("path", ["", "total.", ".amount", "total..amount"])
def test_a_path_with_an_empty_part_is_refused(path):
    with pytest.raises(ValueError, match="is not valid"):
        split_path(path)
