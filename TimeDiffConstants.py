DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"
PRICE_TRESHOLD = 100_000    # for outliers
WEIGHT_TRESHOLD = 50        # for outliers
NUM_OF_HOURS = 24
SEED = 42

COLS_TO_DROP_ALWAYS = ("delivery_timestamp",
                       "session_id",
                       "purchase_id",
                       "event_type",
                       "name",
                       "user_id",
                       'offered_discount',
                       'optional_attributes',
                       'purchase_timestamp')
