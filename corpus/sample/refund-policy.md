# Refund Policy

We want you to be happy with your purchase. This page explains how refunds work.

## International Orders

International orders are refunded to the original payment method within 30 days
of the return being received at our warehouse.

### Exceptions

Customized or made-to-order items are not eligible for a refund unless they
arrive damaged or defective.

## Domestic Orders

Domestic refunds are processed within 5 business days.

## Shipping Cost Table

| Region        | Standard | Express |
|---------------|----------|---------|
| United States | $5.00    | $15.00  |
| Canada        | $8.00    | $20.00  |
| European Union| $12.00   | $28.00  |

## Sample API Response

When you check a refund's status through our API, a completed refund looks
like this:

```json
{
  "refund_id": "rf_9182",
  "status": "completed",
  "amount_cents": 4200,
  "currency": "usd"
}
```

If the status is `pending`, allow up to two additional business days before
contacting support.
