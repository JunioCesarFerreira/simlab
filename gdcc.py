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

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python gdcc.py <número_de_containers>")
        sys.exit(1)
    try:
        n = int(sys.argv[1])
        if n <= 0:
            raise ValueError
        generate_cooja_services(n)
    except ValueError:
        print("Erro: forneça um número inteiro positivo.")
        sys.exit(1)
