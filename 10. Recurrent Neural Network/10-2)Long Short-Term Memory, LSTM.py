
#1) Limit of Vanila RNN->The problem of Long-Term Dependencies

#2) Theory of Vanila RNN->h=tanh(Wx+Wh+b)_은닉상태

#3) LSTM(Long Short-Term Memory) 메모리셀에 입력게이트, 망각게이트, 출력게이트 추가, 셀상태 추가
#입력게이트_현재정보 기억*현재시점의 입력을 얼마나 반영할지*
#삭제게이트_기억을 삭제. 0과가까울수록 많이 삭제, 1과 가까울수록 온전히 기억*이전의 기억을 얼마나 반영할지*
#셀상태(장기상태)_입력게이트에서 선택된 기억을 삭제게이트의 결과값과 더함
#출력 게이트와 은닉 상태(단기 상태)
#nn.RNN(input_dim, hidden_size, batch_first=True) -->> nn.LSTM(input_dim, hidden_size, batch_first=True)
