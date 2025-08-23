import sys

def generate_cooja_services(n):
    base_port = 2231
    for i in range(1, n + 1):
        port = base_port + (i - 1)
        print(f"""
  cooja{i}:
    image: juniocesarferreira/simulation-cooja:v1.1
    container_name: cooja{i}
    restart: unless-stopped
    ports:
      - "{port}:22"
    networks:
      - cooja-net
""".rstrip())

# Generate a docker-compose code for many Cooja Containers
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Use: python gdcc.py <number_of_containers>")
        sys.exit(1)
    try:
        n = int(sys.argv[1])
        if n <= 0:
            raise ValueError
        generate_cooja_services(n)
    except ValueError:
        print("Error: provide a positive integer.")
        sys.exit(1)
