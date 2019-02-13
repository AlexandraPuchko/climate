import sqlite3


def create_database(db_name):
    conn = sqlite3.connect(db_name) # create DB or load if it already exists
    # get cursor to make queries, then add necessary tables if they don't exist
    cur = conn.cursor()
    cur.execute('create table if not exists experiments (exp_id integer, layers integer, \
               layersizes text, lr real, epochs integer)')

    cur.execute('create table if not exists results (exp_id integer, epoch integer, \
               train_loss real, val_loss real)')
    cur.connection.commit()
    # get largest experiment ID from DB
    cur.execute('select max(exp_id) from experiments')
    exp_id = cur.fetchone()[0]
    exp_id = 0 if exp_id is None else exp_id + 1

    return cur, conn, exp_id



def insert_exps(cur, exp_id, layers_sizes, epochs, lr):
    # insert new experiment info into DB
    cur.execute('insert into experiments values (?, ?, ?, ?, ?)',
      (exp_id, len(layers_sizes), ' '.join(map(str, layers_sizes)), lr, epochs))
    cur.connection.commit()

    return



def insert_results(cur, exp_id, epoch, train_loss, dev_loss):
    # insert new experiment info into DB
    cur.execute('insert into results values (?, ?, ?, ?, ?)',
      (exp_id, epoch, train_loss, dev_loss))
    cur.connection.commit()

    return
