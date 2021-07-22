



def did_i_win():
	i_won_a_game = False
	for number_of_games in range(0,100000):
		if number_of_games == 99999:
			i_won_a_game = True
	return i_won_a_game




if __name__ == '__main__':
	did_i_win()