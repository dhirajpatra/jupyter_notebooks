import click

# This is a minimal command-line tool
@click.command()
# This is a command-line option
@click.option('--count', default=1)
def hello(count):
    for x in range(count):
        click.echo('Hello World!')

if __name__ == '__main__':
    hello()