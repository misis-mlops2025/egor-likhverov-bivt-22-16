def sum_numbers(*args):
    return sum(args)


def main():
    result = sum_numbers(7, 10, 52)
    print(f"Сумма чисел: {result}")


if __name__ == "__main__":
    main()
