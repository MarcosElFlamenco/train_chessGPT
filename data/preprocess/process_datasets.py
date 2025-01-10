{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#!/usr/bin/env python3\n",
    "\n",
    "import os\n",
    "import argparse\n",
    "import logging\n",
    "\n",
    "def parse_arguments():\n",
    "    \"\"\"\n",
    "    Parses command-line arguments using argparse.\n",
    "    \"\"\"\n",
    "    parser = argparse.ArgumentParser(\n",
    "        description=\"Parse a large binary file to identify games separated by integer 15, \"\n",
    "                    \"and calculate the total number of games along with their min, max, and average lengths.\"\n",
    "    )\n",
    "    \n",
    "    parser.add_argument(\n",
    "        \"input_file\",\n",
    "        type=str,\n",
    "        help=\"Path to the input binary file.\"\n",
    "    )\n",
    "    \n",
    "    parser.add_argument(\n",
    "        \"--chunk_size\",\n",
    "        type=int,\n",
    "        default=1024,\n",
    "        help=\"Size of each read chunk in bytes. Default is 1024.\"\n",
    "    )\n",
    "    \n",
    "    parser.add_argument(\n",
    "        \"--debug\",\n",
    "        action='store_true',\n",
    "        help=\"Enable debug mode for verbose output.\"\n",
    "    )\n",
    "    \n",
    "    return parser.parse_args()\n",
    "\n",
    "def setup_logging(debug_mode):\n",
    "    \"\"\"\n",
    "    Sets up logging configuration.\n",
    "    \"\"\"\n",
    "    if debug_mode:\n",
    "        logging_level = logging.DEBUG\n",
    "    else:\n",
    "        logging_level = logging.INFO\n",
    "    \n",
    "    logging.basicConfig(\n",
    "        level=logging_level,\n",
    "        format='%(asctime)s - %(levelname)s - %(message)s'\n",
    "    )\n",
    "\n",
    "def process_binary_file(input_file, chunk_size):\n",
    "    \"\"\"\n",
    "    Processes the binary file to identify games and calculate statistics.\n",
    "    \n",
    "    Parameters:\n",
    "        input_file (str): Path to the binary file.\n",
    "        chunk_size (int): Number of bytes to read per chunk.\n",
    "    \n",
    "    Returns:\n",
    "        tuple: (total_games, min_length, max_length, average_length)\n",
    "    \"\"\"\n",
    "    total_games = 0\n",
    "    min_length = None\n",
    "    max_length = None\n",
    "    total_length = 0\n",
    "    \n",
    "    current_game = []\n",
    "    buffer = b''  # To handle cases where 15 is split across chunks\n",
    "    \n",
    "    try:\n",
    "        with open(input_file, 'rb') as f:\n",
    "            while True:\n",
    "                chunk = f.read(chunk_size)\n",
    "                if not chunk:\n",
    "                    break  # End of file\n",
    "                \n",
    "                logging.debug(f\"Read chunk of size: {len(chunk)} bytes\")\n",
    "                \n",
    "                # Prepend any buffered bytes from the previous chunk\n",
    "                data = buffer + chunk\n",
    "                \n",
    "                # Find all indices where byte 15 occurs\n",
    "                indices = [i for i, byte in enumerate(data) if byte == 15]\n",
    "                logging.debug(f\"Found {len(indices)} occurrences of integer 15 in the current chunk\")\n",
    "                \n",
    "                # If there are no 15s in the data, append all to current_game and continue\n",
    "                if not indices:\n",
    "                    current_game.extend(data)\n",
    "                    continue\n",
    "                \n",
    "                # Iterate through all found 15s\n",
    "                for idx, pos in enumerate(indices):\n",
    "                    logging.debug(f\"Processing occurrence {idx+1} of integer 15 at position {pos}\")\n",
    "                    # Start of a new game\n",
    "                    if pos < len(data) - 1:\n",
    "                        # If this is not the first 15, end the previous game\n",
    "                        if current_game:\n",
    "                            game_length = len(current_game)\n",
    "                            total_games += 1\n",
    "                            logging.debug(f\"Ending previous game #{total_games} with length {game_length}\")\n",
    "                            # Update statistics\n",
    "                            if min_length is None or game_length < min_length:\n",
    "                                min_length = game_length\n",
    "                            if max_length is None or game_length > max_length:\n",
    "                                max_length = game_length\n",
    "                            total_length += game_length\n",
    "                            # Reset current_game\n",
    "                            current_game = []\n",
    "                        \n",
    "                        # Start new game with 15\n",
    "                        current_game.append(15)\n",
    "                        logging.debug(f\"Starting new game #{total_games + 1}\")\n",
    "                        \n",
    "                        # If this is the last 15 in the chunk, buffer the remaining data\n",
    "                        if idx == len(indices) - 1:\n",
    "                            # Everything after this pos might belong to the next game\n",
    "                            buffer = data[pos+1:]\n",
    "                            logging.debug(f\"Buffering {len(buffer)} bytes for the next chunk\")\n",
    "                        else:\n",
    "                            # Next 15 found, so everything between current pos and next pos belongs to this game\n",
    "                            next_pos = indices[idx + 1]\n",
    "                            game_data = data[pos+1:next_pos]\n",
    "                            current_game.extend(game_data)\n",
    "                    \n",
    "                # After processing all 15s, check if any data remains after the last 15\n",
    "                last_pos = indices[-1]\n",
    "                if last_pos < len(data) - 1:\n",
    "                    # Buffer the data after the last 15 for the next chunk\n",
    "                    buffer = data[last_pos+1:]\n",
    "                    logging.debug(f\"Buffering {len(buffer)} bytes after the last 15\")\n",
    "                else:\n",
    "                    buffer = b''\n",
    "                \n",
    "        # After reading all chunks, handle the last game if exists\n",
    "        if current_game:\n",
    "            game_length = len(current_game)\n",
    "            total_games += 1\n",
    "            logging.debug(f\"Ending last game #{total_games} with length {game_length}\")\n",
    "            # Update statistics\n",
    "            if min_length is None or game_length < min_length:\n",
    "                min_length = game_length\n",
    "            if max_length is None or game_length > max_length:\n",
    "                max_length = game_length\n",
    "            total_length += game_length\n",
    "    \n",
    "    except FileNotFoundError:\n",
    "        logging.error(f\"File not found: {input_file}\")\n",
    "        return (0, 0, 0, 0)\n",
    "    except Exception as e:\n",
    "        logging.error(f\"An error occurred while processing the file: {e}\")\n",
    "        return (total_games, min_length if min_length else 0, \n",
    "                max_length if max_length else 0, \n",
    "                (total_length / total_games) if total_games > 0 else 0)\n",
    "    \n",
    "    # Calculate average length\n",
    "    average_length = (total_length / total_games) if total_games > 0 else 0\n",
    "    \n",
    "    return (total_games, min_length if min_length else 0, \n",
    "            max_length if max_length else 0, \n",
    "            average_length)\n",
    "\n",
    "def main():\n",
    "    args = parse_arguments()\n",
    "    setup_logging(args.debug)\n",
    "    \n",
    "    logging.info(f\"Starting processing of file: {args.input_file}\")\n",
    "    logging.info(f\"Using chunk size: {args.chunk_size} bytes\")\n",
    "    \n",
    "    total_games, min_length, max_length, average_length = process_binary_file(\n",
    "        args.input_file,\n",
    "        args.chunk_size\n",
    "    )\n",
    "    \n",
    "    logging.info(\"Processing complete.\")\n",
    "    print(\"\\n=== Game Statistics ===\")\n",
    "    print(f\"Total number of games: {total_games}\")\n",
    "    if total_games > 0:\n",
    "        print(f\"Minimum game length: {min_length} integers\")\n",
    "        print(f\"Maximum game length: {max_length} integers\")\n",
    "        print(f\"Average game length: {average_length:.2f} integers\")\n",
    "    else:\n",
    "        print(\"No games found.\")\n",
    "\n",
    "if __name__ == \"__main__\":\n",
    "    main()\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "env",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.12.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
