import logging
import os
import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                    filename=f"output {datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}.log", filemode='w'
                    )

RATING_MULTIPLIER = 5.0