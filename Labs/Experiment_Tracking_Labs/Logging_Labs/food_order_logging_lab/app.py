import logging

# Create custom logger
logger = logging.getLogger("food_order_app")
logger.setLevel(logging.DEBUG)

# Prevent duplicate logs
logger.propagate = False

# Create formatter
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# File handler
file_handler = logging.FileHandler("app.log", mode="w")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

# Add handlers only once
if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


def validate_order(item, quantity):
    logger.debug(f"Validating order: item={item}, quantity={quantity}")

    if not item:
        logger.error("Order validation failed: item name is missing")
        return False

    if quantity <= 0:
        logger.warning(f"Invalid quantity entered for {item}: {quantity}")
        return False

    logger.info(f"Order validated successfully for item: {item}")
    return True


def calculate_total(price, quantity):
    logger.debug(f"Calculating total: price={price}, quantity={quantity}")
    total = price * quantity
    logger.info(f"Total price calculated successfully: {total}")
    return total


def process_payment(total_amount, payment_amount):
    logger.debug(
        f"Processing payment: total_amount={total_amount}, payment_amount={payment_amount}"
    )

    try:
        if payment_amount < total_amount:
            raise ValueError("Insufficient payment amount")

        balance = payment_amount - total_amount
        logger.info(f"Payment processed successfully. Balance: {balance}")
        return balance

    except ValueError:
        logger.exception("Payment processing failed")
        return None


def generate_receipt(item, quantity, total, balance):
    logger.debug("Generating receipt")
    receipt = (
        f"\\n--- RECEIPT ---\\n"
        f"Item: {item}\\n"
        f"Quantity: {quantity}\\n"
        f"Total: ${total}\\n"
        f"Balance Returned: ${balance}\\n"
        f"----------------\\n"
    )
    logger.info("Receipt generated successfully")
    return receipt


def main():
    logger.info("Food order application started")

    item = "Pizza"
    quantity = 2
    price_per_item = 12
    payment_amount = 30

    if validate_order(item, quantity):
        total = calculate_total(price_per_item, quantity)
        balance = process_payment(total, payment_amount)

        if balance is not None:
            receipt = generate_receipt(item, quantity, total, balance)
            print(receipt)

    validate_order("Burger", 0)

    total2 = calculate_total(15, 2)
    process_payment(total2, 10)

    logger.critical("Demo critical log: end of application test")
    logger.info("Food order application finished")


if __name__ == "__main__":
    main()
