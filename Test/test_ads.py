#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
import os

# GPIO numbers for Jetson Nano (sysfs numbering)
# SCK  = GPIO 18 (Pin 23)
# MOSI = GPIO 16 (Pin 19)
# MISO = GPIO 17 (Pin 21)
# CS   = GPIO 194 (Pin 15)
# RST  = GPIO 79 (Pin 12)
# DRDY = GPIO 50 (Pin 11) – not used but exported for completeness

SCK_GPIO   = 18
MOSI_GPIO  = 16
MISO_GPIO  = 17
CS_GPIO    = 194
RST_GPIO   = 79
DRDY_GPIO  = 50

def export_gpio(gpio):
    if not os.path.exists(f"/sys/class/gpio/gpio{gpio}"):
        with open("/sys/class/gpio/export", "w") as f:
            f.write(str(gpio))

def unexport_gpio(gpio):
    with open("/sys/class/gpio/unexport", "w") as f:
        f.write(str(gpio))

def set_direction(gpio, direction):
    with open(f"/sys/class/gpio/gpio{gpio}/direction", "w") as f:
        f.write(direction)

def set_value(gpio, value):
    with open(f"/sys/class/gpio/gpio{gpio}/value", "w") as f:
        f.write(str(value))

def get_value(gpio):
    with open(f"/sys/class/gpio/gpio{gpio}/value", "r") as f:
        return int(f.read().strip())

def delay_us(t):
    time.sleep(t / 1_000_000.0)

def spi_write_byte(byte):
    for bit in range(7, -1, -1):
        set_value(MOSI_GPIO, (byte >> bit) & 1)
        set_value(SCK_GPIO, 1)
        delay_us(1)
        set_value(SCK_GPIO, 0)
        delay_us(1)

def spi_read_byte():
    byte = 0
    for _ in range(8):
        set_value(SCK_GPIO, 1)
        delay_us(1)
        bit = get_value(MISO_GPIO)
        byte = (byte << 1) | bit
        set_value(SCK_GPIO, 0)
        delay_us(1)
    return byte

def main():
    # Export and configure GPIOs
    for gpio in [SCK_GPIO, MOSI_GPIO, MISO_GPIO, CS_GPIO, RST_GPIO, DRDY_GPIO]:
        export_gpio(gpio)
        time.sleep(0.05)

    set_direction(SCK_GPIO, "out")
    set_direction(MOSI_GPIO, "out")
    set_direction(MISO_GPIO, "in")
    set_direction(CS_GPIO, "out")
    set_direction(RST_GPIO, "out")
    set_direction(DRDY_GPIO, "in")

    # Initial values
    set_value(CS_GPIO, 1)
    set_value(RST_GPIO, 1)
    set_value(SCK_GPIO, 0)
    set_value(MOSI_GPIO, 0)

    try:
        print("Resetting ADS1263...")
        # Hardware reset
        set_value(RST_GPIO, 1)
        time.sleep(0.2)
        set_value(RST_GPIO, 0)
        time.sleep(0.2)
        set_value(RST_GPIO, 1)
        time.sleep(0.2)

        # Software reset command
        set_value(CS_GPIO, 0)
        spi_write_byte(0x06)
        set_value(CS_GPIO, 1)
        time.sleep(0.1)

        print("Reading ID register...")
        set_value(CS_GPIO, 0)
        spi_write_byte(0x20)  # RREG | address 0x00
        spi_write_byte(0x00)  # read 1 byte
        id_byte = spi_read_byte()
        set_value(CS_GPIO, 1)

        chip_id = (id_byte >> 5) & 0x07
        print(f"Full byte: 0x{id_byte:02X}, Chip ID: 0x{chip_id:02X}")

        if chip_id == 0x01:
            print("Success! ID is correct (0x01). Software SPI via sysfs works.")
        else:
            print(f"Error: Wrong ID. Expected 0x01, got 0x{chip_id:02X}")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Unexport GPIOs to release them
        for gpio in [SCK_GPIO, MOSI_GPIO, MISO_GPIO, CS_GPIO, RST_GPIO, DRDY_GPIO]:
            try:
                unexport_gpio(gpio)
            except:
                pass

if __name__ == "__main__":
    main()
