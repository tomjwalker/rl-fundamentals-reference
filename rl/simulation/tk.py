"""
Example log lines:
2024-09-16T12:01:35 - pageId: {} - customerId: {}
2024-09-16T12:01:37 - pageId: {} - customerId: {}
2024-09-16T15:11:51 - pageId: {} - customerId: {}
"""


from collections import defaultdict
from collections import Counter

def extract_id(entry, type):
    pass


# Find the loyal customers that visited the website on consecutive days and visited more than one page
def loyal_customers(log_file_day_1: list[str], log_file_day_2: list[str]) -> set[str]:

    customer_logs = defaultdict(set)
    for entry in log_file_day_1:
        page = extract_id(entry, "pageId")
        customer = extract_id(entry, "customerId")
        customer_logs[customer].add(page)

    loyal_customers = set()
    for entry in log_file_day_2:
        page = extract_id(entry, "pageId")
        customer = extract_id(entry, "customerId")
        customer_logs[customer].add(page)
        if len(customer_logs[customer]) > 1:
            loyal_customers.add(customer)

    return loyal_customers

    # If customerId in log_file_day_2 and log_file_day_1

    # And set of all pages they have looked at across 2 days > 1

    # Then loyal customer
    pass


dummy_log = [
    "2024-09-16T12:01:35 - pageId: A - customerId: A",
    "2024-09-16T12:01:37 - pageId: B - customerId: A",
    "2024-09-16T15:11:51 - pageId: C - customerId: B",
    "2024-09-17T12:01:35 - pageId: A - customerId: A",
]




# Find the top ten exit pages that users most often leave the website on
def exit_pages(log_file_day_1: list[str]) -> set[str]:
    customer_logs = dict()
    for entry in log_file_day_1:
        page = extract_id(entry, "pageId")
        customer = extract_id(entry, "customerId")
        customer_logs[customer] = page

    last_pages = Counter()
    for customer, pages in customer_logs:
        last_page = pages
        last_pages.update(last_page)

    return last_pages.most_common(10)

